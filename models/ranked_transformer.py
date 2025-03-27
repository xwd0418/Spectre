import logging
import pytorch_lightning as pl
import torch, pickle
import math
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import torch.distributed as dist



from utils import ranker, constants
from models import compute_metrics
from models.encoders.encoder_factory import build_encoder
from models.extras.transformer_stuff import (
    generate_square_subsequent_mask
)
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
import os

from utils.lr_scheduler import NoamOpt
from utils.L1_decay import L1

import sys, pathlib
repo_path = pathlib.Path(__file__).resolve().parents[1]

RANKED_TNSFMER_ARGS = [
    "dim_model", "dim_coords", "heads", "layers", "ff_dim", "coord_enc", "wavelength_bounds",
    # exclude save_params
    "gce_resolution", "dropout", "weight_decay", "out_dim", "pos_weight", "ranking_set_path", "FP_choice",
    "lr", "noam_factor", "scheduler", "freeze_weights", "bs", "warm_up_steps", "loss_func", 
]


class HsqcRankedTransformer(pl.LightningModule):
    """A Transformer encoder for input HSQC.
    Parameters
    ----------
    """

    def __init__(
        self,
        # model args
        dim_model=128,
        dim_coords=[43, 43, 42],
        heads=8,
        layers=8,
        ff_dim=1024,
        coord_enc="sce",
        wavelength_bounds=None,
        gce_resolution=1,
        dropout=0.1,
        # other business logic stuff
        save_params=True,
        ranking_set_path="",
        FP_choice="R2-6144FP",
        loss_func = "",
        # training args
        lr=1e-5,
        noam_factor = 1,
        pos_weight = None,
        weight_decay = 0.0,
        L1_decay = 0.0,
        scheduler=None,  # None, "attention"
        warm_up_steps=0,
        freeze_weights=False,
        use_Jaccard = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        
        from datasets.dataset_utils import fp_loader_configer
        self.fp_loader = fp_loader_configer.fp_loader
        
        params = locals().copy()
        self.out_logger = logging.getLogger("lightning")
        if dist.is_initialized():
            rank = dist.get_rank()
            if rank != 0:
                # For any process with rank other than 0, set logger level to WARNING or higher
                self.out_logger.setLevel(logging.WARNING)
        self.out_logger.info("[RankedTransformer] Started Initializing")
        self.logger_should_sync_dist = torch.cuda.device_count() > 1
        # logging print
        for k, v in params.items():
            if k in RANKED_TNSFMER_ARGS:
                self.out_logger.info(f"[RankedTransformer] {k=},{v=}")


        # === All Parameters ===
        out_dim = kwargs['out_dim']         
        self.FP_length = out_dim # 6144 
        self.separate_classifier = kwargs['separate_classifier']
        self.test_on_deepsat_retrieval_set = kwargs['test_on_deepsat_retrieval_set']
        if FP_choice == "R0_to_R4_30720_FP":
            out_dim = self.FP_length * 5
        elif FP_choice == "R0_to_R6_exact_R_concat_FP":
            out_dim = self.FP_length * 7
        if loss_func == "CE":
            assert(FP_choice == "R2-6144-count-based-FP")
            out_dim = self.FP_length * kwargs['num_class']
        self.out_dim = out_dim
        
        self.bs = kwargs['bs']
        self.num_class = kwargs['num_class'] if loss_func == "CE" else None
        self.lr = lr
        self.noam_factor = noam_factor
        self.weight_decay = weight_decay

        self.scheduler = scheduler
        self.warm_up_steps = warm_up_steps
        self.dim_model = dim_model
        
        self.use_Jaccard = use_Jaccard
        
        # don't set ranking set if you just want to treat it as a module
        self.FP_choice=FP_choice
        self.rank_by_soft_output = kwargs['rank_by_soft_output']
        self.rank_by_test_set = kwargs['rank_by_test_set']
        
        
        if FP_choice.startswith("DB_specific_FP") :
            self.radius = int(FP_choice.split("_")[-1])
        # elif FP_choice.startswith("pick_entropy"):
        #     raise NotImplementedError("pick_entropy getting radius is not implemented")
        
        if not self.rank_by_test_set:
            self.change_ranker_for_inference()
        else:
            assert FP_choice=="HYUN_FP" or FP_choice.startswith("DB_specific_FP") or FP_choice.startswith("pick_entropy"), "rank_by_test_set is True, but received unexpected FP_choice"
            self.ranker = ranker.RankingSet(store=self.fp_loader.build_rankingset("val", predefined_FP = FP_choice),
                                             batch_size=self.bs, CE_num_class=self.num_class)

        if save_params:
            print("HsqcRankedTransformer saving args")
            self.save_hyperparameters(*RANKED_TNSFMER_ARGS, *kwargs.keys())

        # ranked encoder
        self.enc = build_encoder(
            coord_enc, dim_model, dim_coords, wavelength_bounds, gce_resolution, kwargs['use_peak_values'])
        self.out_logger.info(
            f"[RankedTransformer] Using {str(self.enc.__class__)}")


        ### Loss function 
        if pos_weight==None:
            self.bce_pos_weight = None
            self.out_logger.info("[RankedTransformer] bce_pos_weight = None")
        else:
            try:
                pos_weight_value = float(pos_weight)
                # self.bce_pos_weight= torch.full((self.FP_length,), pos_weight_value)
                self.bce_pos_weight= torch.tensor([pos_weight_value])
                self.out_logger.info(f"[RankedTransformer] bce_pos_weight is {pos_weight_value}")
        
            except :
                if pos_weight == "ratio":
                    self.bce_pos_weight = torch.load(f'{repo_path}/pos_weight_array_based_on_ratio.pt')
                    self.out_logger.info("[RankedTransformer] bce_pos_weight is loaded ")
                else:
                    raise ValueError(f"pos_weight {pos_weight} is not valid")
        
        self.loss_func = loss_func
        if FP_choice == "R2-6144-count-based-FP":
            if loss_func == "MSE":
                self.loss = nn.MSELoss()
                self.compute_metric_func = compute_metrics.cm_count_based_mse
            elif loss_func == "CE":
                self.loss = nn.CrossEntropyLoss()
                self.compute_metric_func = compute_metrics.cm_count_based_ce
            else:
                raise Exception("loss_func should be either MSE or CE when using count-based FP")
        else: #  Bit based FP
            self.loss = nn.BCEWithLogitsLoss(pos_weight=self.bce_pos_weight)
            self.compute_metric_func = compute_metrics.cm
        
        
        
        # additional nn modules 
        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.test_step_outputs = []
        from collections import defaultdict
        self.test_np_classes_rank1 = defaultdict(list)

        self.NMR_type_embedding = nn.Embedding(4, dim_model)
        # HSQC, C NMR, H NMR, MW
        # MW isn't NMR, but, whatever......
        self.out_logger.info("[RankedTransformer] nn.linear layer to be initialized")
        # print("out_dim is ", out_dim, " dim_model is ", dim_model)
        # exit(0)
        self.fc = nn.Linear(dim_model, out_dim)
        self.out_logger.info("[RankedTransformer] nn.linear layer is initialized")
        # (1, 1, dim_model)
        self.latent = torch.nn.Parameter(torch.randn(1, 1, dim_model)) # the <cls> token

        # The Transformer layers:
        layer = torch.nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=layers,
        )
        # === END Parameters ===

        self.out_logger.info("[RankedTransformer] weights are initialized")
        
        if L1_decay:
            self.out_logger.info("[RankedTransformer] L1_decay is applied")
            self.transformer_encoder = L1(self.transformer_encoder, L1_decay)
            self.fc = L1(self.fc, L1_decay)
            
        if freeze_weights:
            self.out_logger.info("[RankedTransformer] Freezing Weights")
            for parameter in self.parameters():
                parameter.requires_grad = False
        self.out_logger.info("[RankedTransformer] Initialized")

    @staticmethod
    def add_model_specific_args(parent_parser, model_name=""):
        model_name = model_name if len(model_name) == 0 else f"{model_name}_"
        parser = parent_parser.add_argument_group(model_name)
        parser.add_argument(f"--{model_name}lr", type=float, default=1e-5)
        parser.add_argument(f"--{model_name}noam_factor", type=float, default=1.0)
        parser.add_argument(f"--{model_name}dim_model", type=int, default=784)
        parser.add_argument(f"--{model_name}dim_coords", metavar='N',
                            type=int, default=[365, 365, 54 ],
                            nargs="+", action="store")
        parser.add_argument(f"--{model_name}heads", type=int, default=8)
        parser.add_argument(f"--{model_name}layers", type=int, default=16)
        parser.add_argument(f"--{model_name}ff_dim", type=int, default=3072)
        parser.add_argument(f"--{model_name}wavelength_bounds",
                            type=float, default=[[0.01, 400.0], [0.01, 20.0]], nargs='+', action='append')
        parser.add_argument(f"--{model_name}dropout", type=float, default=0.1)
        parser.add_argument(f"--{model_name}pos_weight", type=str, default=None, 
                            help = "if none, then not to be used; if ratio,\
                                then used the save tensor which is the ratio of num_0/num_1, \
                                if float num ,then use this as the ratio")
        parser.add_argument(f"--{model_name}weight_decay", type=float, default=0.0)
        parser.add_argument(f"--{model_name}L1_decay", type=float, default=0.0)
        parser.add_argument(f"--{model_name}warm_up_steps", type=int, default=8000)
        parser.add_argument(f"--{model_name}scheduler", type=str, default="attention")
        parser.add_argument(f"--{model_name}coord_enc", type=str, default="sce")
        parser.add_argument(
            f"--{model_name}gce_resolution", type=float, default=1)
        parser.add_argument(f"--{model_name}freeze_weights",
                            type=bool, default=False)
        parser.add_argument(
            f"--{model_name}ranking_set_path", type=str, default="")
        return parent_parser

    @staticmethod
    def prune_args(vals: dict, model_name=""):
        items = [(k[len(model_name) + 1:], v)
                 for k, v in vals.items() if k.startswith(model_name)]
        return dict(items)

    def encode(self, hsqc, NMR_type_indicator, mask=None):
        """
        Returns
        -------
        latent : torch.Tensor of shape (n_spectra, n_peaks + 1, dim_model)
            The latent representations for the spectrum and each of its
            peaks.
        mem_mask : torch.Tensor
            The memory mask specifying which elements were padding in X.
        """
        if mask is None:
            zeros = ~hsqc.sum(dim=2).bool()
            mask = [
                torch.tensor([[False]] * hsqc.shape[0]).type_as(zeros),
                zeros,
            ]
            mask = torch.cat(mask, dim=1)
            mask = mask.to(self.device)

        points = self.enc(hsqc) # something like positional encoding , but encoding cooridinates
        NMR_type_embedding = self.NMR_type_embedding(NMR_type_indicator)
        # print("points shape is ", points.shape, " NMR_type_embedding shape is ", NMR_type_embedding.shape)
        points += NMR_type_embedding
        # print(points.shape)
        # Add the spectrum representation to each input:
        latent = self.latent.expand(points.shape[0], -1, -1) # make batch_size copies of latent
        # print(latent.device, points.device)
        
        points = torch.cat([latent, points], dim=1)
      
        out = self.transformer_encoder(points, src_key_padding_mask=mask)
        return out, mask
    
    def forward(self, hsqc, NMR_type_indicator, return_representations=False):
        """The forward pass.
        Parameters
        ----------
        hsqc: torch.Tensor of shape (batch_size, n_points, 3)
            The hsqc to embed. Axis 0 represents an hsqc, axis 1
            contains the coordinates in the hsqc, and axis 2 is essentially is
            a 3-tuple specifying the coordinate's x, y, and z value. These
            should be zero-padded, such that all of the hsqc in the batch
            are the same length.
        """
        out, _ = self.encode(hsqc, NMR_type_indicator)  # (b_s, seq_len, dim_model)
        out_cls = self.fc(out[:, :1, :].squeeze(1))  # extracts cls token : (b_s, dim_model) -> (b_s, out_dim)
        if return_representations:
            return out.detach().cpu().numpy()
        return out_cls

    def training_step(self, batch, batch_idx):
        
        inputs, labels, NMR_type_indicator = batch
        out = self.forward(inputs, NMR_type_indicator)
        if self.loss_func == "CE":
            out = out.view(out.shape[0],  self.num_class, self.FP_length)

        loss = self.loss(out, labels)
        
        self.log("tr/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, NMR_type_indicator = batch
        out = self.forward(inputs, NMR_type_indicator)
        if self.loss_func == "CE":
            out = out.view(out.shape[0],  self.num_class, self.FP_length)
            preds = out.argmax(dim=1)
        else:
            preds = out
        # print((labels) )
        # print("\n\n\n")
        loss = self.loss(out, labels)
        metrics, rank_1_hits = self.compute_metric_func(
            preds, labels, self.ranker, loss, self.loss, thresh=0.0, 
            rank_by_soft_output=self.rank_by_soft_output,
            query_idx_in_rankingset=batch_idx,
            use_Jaccard = self.use_Jaccard
            )
        if type(self.validation_step_outputs)==list: # adapt for child class: optional_input_ranked_transformer
            self.validation_step_outputs.append(metrics)
        return metrics
    
    def test_step(self, batch, batch_idx):
        inputs, labels, NMR_type_indicator, np_classes = batch
        out = self.forward(inputs, NMR_type_indicator)
        if self.loss_func == "CE":
            out = out.view(out.shape[0],  self.num_class, self.FP_length)
            preds = out.argmax(dim=1)
        else:
            preds = out
        loss = self.loss(out, labels)
        metrics, rank_1_hits = self.compute_metric_func(
            preds, labels, self.ranker, loss, self.loss, thresh=0.0,
            rank_by_soft_output=self.rank_by_soft_output,
            query_idx_in_rankingset=batch_idx,
            use_Jaccard = self.use_Jaccard
            )
        
        if type(self.test_step_outputs)==list:
            self.test_step_outputs.append(metrics)
            for curr_classes, curr_rank_1_hits in zip(np_classes, rank_1_hits.tolist()):
                # print(curr_rank_1_hits)
                for np_class in curr_classes:
                    self.test_np_classes_rank1[np_class].append(curr_rank_1_hits)
        return metrics, np_classes, rank_1_hits

    def predict_step(self, batch, batch_idx, return_representations=False):
        x, smiles_chemical_name = batch
        # smiles, names = zip(*smiles_chemical_name)
        # print(smiles, names)
        if return_representations:
            return self.forward(x, return_representations=True)
        out = self.forward(x)
        preds = torch.sigmoid(out)
        top_k_idxs = self.ranker.retrieve_idx(preds)
        return top_k_idxs
            
    # def on_train_start(self, trainer, pl_module):
    #     if dist.is_initialized():
    #         rank = dist.get_rank()
    #         if rank != 0:
    #             # For any process with rank other than 0, set logger level to WARNING or higher
    #             self.out_logger.setLevel(logging.WARNING)
        
    def on_train_epoch_end(self):
        # return
        if self.training_step_outputs:
            feats = self.training_step_outputs[0].keys()
            di = {}
            for feat in feats:
                di[f"tr/mean_{feat}"] = np.mean([v[feat]
                                                for v in self.training_step_outputs])
            for k, v in di.items():
                self.log(k, v, on_epoch=True)
            self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        # return
        feats = self.validation_step_outputs[0].keys()
        di = {}
        for feat in feats:
            di[f"val/mean_{feat}"] = np.mean([v[feat]
                                             for v in self.validation_step_outputs])
        for k, v in di.items():
            self.log(k, v, on_epoch=True, prog_bar=k=="val/mean_rank_1")
        self.validation_step_outputs.clear()
        
    def on_test_epoch_end(self):
        feats = self.test_step_outputs[0].keys()
        di = {}
        for feat in feats:
            di[f"test/mean_{feat}"] = np.mean([v[feat]
                                             for v in self.test_step_outputs])
        for k, v in di.items():
            self.log(k, v, on_epoch=True)
            # self.log(k, v, on_epoch=True)
            
        for np_class, rank1_hits in self.test_np_classes_rank1.items():
            self.log(f"test/rank_1_of_NP_class/{np_class}", np.mean(rank1_hits), on_epoch=True)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        if not self.scheduler:
            return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay = self.weight_decay)
        elif self.scheduler == "attention":
            optim = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay = self.weight_decay, 
                                     betas=(0.9, 0.98), eps=1e-9)
            
            scheduler = NoamOpt(self.dim_model, self.warm_up_steps, optim, self.noam_factor)
            
            return {
                "optimizer": optim,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            }

    def log(self, name, value, *args, **kwargs):
        # Set 'sync_dist' to True by default
        if kwargs.get('sync_dist') is None:
            kwargs['sync_dist'] = kwargs.get(
                'sync_dist', self.logger_should_sync_dist)
        if name == "test/mean_rank_1":
            print(kwargs,"\n\n")
        super().log(name, value, *args, **kwargs)
        
    def change_ranker_for_testing(self):
        if self.rank_by_test_set:
            self.ranker = ranker.RankingSet(store=self.fp_loader.build_rankingset("test", predefined_FP = self.FP_choice),
                                            batch_size=self.bs, CE_num_class=self.num_class)
            
        # else: # keep the same ranker

    def change_ranker_for_inference(self,):
        use_hyun_fp = self.FP_choice=="HYUN_FP"
        self.ranker = ranker.RankingSet(store=self.fp_loader.build_inference_ranking_set_with_everything(
                                                                fp_dim = self.FP_length, 
                                                                max_radius = self.radius,
                                                                use_hyun_fp=use_hyun_fp,
                                                                test_on_deepsat_retrieval_set=self.test_on_deepsat_retrieval_set),
                                          batch_size=self.bs, CE_num_class=self.num_class, need_to_normalize=False, )

















# MOONSHOT_ARGS = [
#     "pad_token_idx", "vocab_size", "d_model", "num_layers", "num_heads", "d_feedforward",
#     "activation", "max_seq_len", "dropout"
# ]


# class Moonshot(HsqcRankedTransformer):
#     """
#       Only parameters, no sampling
#     """

#     def __init__(self,
#                  pad_token_idx,
#                  vocab_size,
#                  d_model,
#                  num_layers,
#                  num_heads,
#                  d_feedforward,
#                  # lr,
#                  # weight_decay,
#                  activation,
#                  # num_steps,
#                  max_seq_len,
#                  # schedule,
#                  # warm_up_steps,
#                  tokeniser,
#                  dropout=0.1,
#                  *args,
#                  **kwargs):
#         super().__init__(save_params=False, *args, **kwargs)
#         moonshot_params = locals()
#         moonshot_args = {k: moonshot_params[k] for k in MOONSHOT_ARGS}
#         ranked_tnsfm_args = {k: kwargs[k]
#                              for k in RANKED_TNSFMER_ARGS if k in kwargs}
#         to_save = {**ranked_tnsfm_args, **moonshot_args}
#         print(to_save)
#         self.save_hyperparameters(to_save, logger=True)

#         # _AbsTransformerModel
#         self.pad_token_idx = pad_token_idx
#         self.vocab_size = vocab_size
#         self.d_model = d_model
#         self.num_layers = num_layers
#         self.num_heads = num_heads
#         self.d_feedforward = d_feedforward
#         self.activation = activation
#         self.dropout = nn.Dropout(dropout)
#         self.max_seq_len = max_seq_len
#         self.tokeniser = tokeniser

#         # BART stuff
#         self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_idx)
#         dec_norm = nn.LayerNorm(d_model)
#         dec_layer = PreNormDecoderLayer(
#             d_model, num_heads, d_feedforward, dropout, activation)
#         self.decoder = nn.TransformerDecoder(
#             dec_layer, num_layers, norm=dec_norm)
#         self.token_fc = nn.Linear(d_model, vocab_size)
#         self.loss_fn = nn.CrossEntropyLoss(
#             reduction="none", ignore_index=pad_token_idx)
#         self.log_softmax = nn.LogSoftmax(dim=2)
#         self.register_buffer("pos_emb", self._positional_embs())

#     # Ripped from chemformer
#     def _construct_input(self, token_ids):
#         """
#           Expects tokens in (seq_len, b_s) format

#         Returns:
#           (seq_len, b_s, d_model) embedding (with dropout applied)
#         """
#         seq_len, _ = tuple(token_ids.size())
#         token_embs = self.emb(token_ids)

#         # Scaling the embeddings like this is done in other transformer libraries
#         token_embs = token_embs * math.sqrt(self.d_model)

#         positional_embs = self.pos_emb[:seq_len,
#                                        :].unsqueeze(0).transpose(0, 1)
#         embs = token_embs + positional_embs
#         embs = self.dropout(embs)
#         return embs

#     # Ripped from chemformer
#     def _positional_embs(self):
#         """ Produces a tensor of positional embeddings for the model

#         Returns a tensor of shape (self.max_seq_len, self.d_model) filled with positional embeddings,
#         which are created from sine and cosine waves of varying wavelength
#         """

#         encs = torch.tensor(
#             [dim / self.d_model for dim in range(0, self.d_model, 2)])
#         encs = 10000 ** encs
#         encs = [(torch.sin(pos / encs), torch.cos(pos / encs))
#                 for pos in range(self.max_seq_len)]
#         encs = [torch.stack(enc, dim=1).flatten()[:self.d_model]
#                 for enc in encs]
#         encs = torch.stack(encs)
#         return encs

#     def sample(self, hsqc):
#         """
#           hsqc: (bs, seq_len)
#         """
#         bs, _, _ = hsqc.size()
#         pad_token_idx = 0
#         begin_token_idx = 2
#         end_token_idx = 3
#         with torch.no_grad():
#             # (bs, seq_len)
#             tokens = torch.ones((bs, self.max_seq_len), dtype=torch.int64).to(
#                 self.device) * pad_token_idx
#             tokens[:, 0] = begin_token_idx
#             # (bs, seq_len)
#             pad_mask = torch.zeros((bs, self.max_seq_len),
#                                    dtype=torch.bool).to(self.device)

#             memory, encoder_mask = self.encode(hsqc)

#             for i in range(1, self.max_seq_len):
#                 decoder_inputs = tokens[:, :i]
#                 decoder_mask = pad_mask[:, :i]
#                 # (seq_len, bs, vocab_size) for token_output
#                 model_output = self.decode(
#                     memory, encoder_mask, decoder_inputs, decoder_mask)["token_output"]
#                 best_ids = torch.argmax(
#                     model_output[[-1], :, :], dim=2).squeeze(0).long()  # (bs)

#                 tokens[:, i] = best_ids
#                 pad_mask[:, i] = (best_ids == end_token_idx) | (
#                     best_ids == pad_token_idx)

#                 if torch.all(pad_mask):
#                     break
#             # (bs, seq_len)
#             my_tokens = tokens.transpose(0, 1).tolist()
#             str_tokens = self.tokeniser.convert_ids_to_tokens(my_tokens)
#             mol_strs = self.tokeniser.detokenise(str_tokens)
#             return mol_strs

#     def sample_rdm(self, hsqc, temperature=1.0, gen_len=None):
#         """
#           hsqc: (bs, seq_len)
#         """
#         bs, _, _ = hsqc.size()
#         pad_token_idx = 0
#         begin_token_idx = 2
#         end_token_idx = 3

#         max_len = self.max_seq_len
#         if gen_len:
#             max_len = gen_len

#         with torch.no_grad():
#             # (bs, seq_len)
#             tokens = torch.ones((bs, gen_len), dtype=torch.int64).to(
#                 self.device) * pad_token_idx
#             tokens[:, 0] = begin_token_idx
#             # (bs, seq_len)
#             pad_mask = torch.zeros((bs, gen_len),
#                                    dtype=torch.bool).to(self.device)
#             print(f"Max seq len: {gen_len}")
#             memory, encoder_mask = self.encode(hsqc)

#             for i in tqdm.tqdm(range(1, gen_len)):
#                 decoder_inputs = tokens[:, :i]
#                 decoder_mask = pad_mask[:, :i]
#                 # (seq_len, bs, vocab_size) for token_output
#                 model_output = self.decode(
#                     memory, encoder_mask, decoder_inputs, decoder_mask)["token_output"]
#                 # (seq_len, bs, vocab_size)
#                 probability_output = F.softmax(
#                     model_output / temperature, dim=2)
#                 sampled_ids = torch.multinomial(
#                     probability_output[-1, :, :], num_samples=1).flatten()
#                 tokens[:, i] = sampled_ids
#                 pad_mask[:, i] = (sampled_ids == end_token_idx) | (
#                     sampled_ids == pad_token_idx)

#                 if torch.all(pad_mask):
#                     break

#             # (bs, seq_len)
#             my_tokens = tokens.tolist()
#             str_tokens = self.tokeniser.convert_ids_to_tokens(my_tokens)
#             mol_strs = self.tokeniser.detokenise(str_tokens)
#             return {
#                 "mol_strs": mol_strs,
#                 "token_ids": my_tokens,
#                 "tokens": str_tokens,
#             }

#     def decode(self, memory, encoder_mask, decoder_inputs, decoder_mask):
#         """

#         Args:
#             memory: (b_s, seq_len, dim_model)
#             encoder_padding_mask : (b_s, seq_len)
#             decoder_inputs: (b_s, seq_len)
#             decoder_mask: (b_s, seq_len)

#         Returns:
#             {
#               model_output: (s_l, b_s, d_model)
#               token_output: (s_l, b_s, vocab_size)
#             }
#         """
#         _, s_l = decoder_mask.size()

#         # (s_l, s_l)
#         tgt_mask = generate_square_subsequent_mask(s_l, device=self.device)
#         # (b_s, s_l, dim)
#         decoder_embs = self._construct_input(decoder_inputs)

#         # embs, memory need seq_len, batch_size convention
#         # (s_l, b_s, dim), (s_l, b_s, dim)
#         decoder_embs, memory = decoder_embs.transpose(
#             0, 1), memory.transpose(0, 1)

#         model_output = self.decoder(
#             decoder_embs,
#             memory,
#             tgt_mask=tgt_mask,  # prevent cheating mask
#             tgt_key_padding_mask=decoder_mask,  # padding mask
#             memory_key_padding_mask=encoder_mask  # padding mask
#         )

#         token_output = self.token_fc(model_output)

#         return {
#             "model_output": model_output,
#             "token_output": token_output,
#         }

#     def forward(self, batch):
#         # I use (batch_size, seq_len convention)
#         # see datasets/dataset_utils.py:tokenise_and_mask
#         hsqc, collated_smiles = batch

#         # encoder
#         # (b_s, seq_len, dim_model), (b_s, seq_len)
#         memory, encoder_mask = self.encode(hsqc)

#         # decode
#         decoder_inputs = collated_smiles["decoder_inputs"]
#         decoder_mask = collated_smiles["decoder_mask"]

#         return self.decode(memory, encoder_mask, decoder_inputs, decoder_mask)

#     def _calc_loss(self, batch_input, model_output):
#         """ Calculate the loss for the model

#         Args:
#             batch_input (dict): Input given to model,
#             model_output (dict): Output from model

#         Returns:
#             loss (singleton tensor),
#         """

#         target = batch_input["target"]  # (b_s, s_l)
#         target_mask = batch_input["target_mask"]  # (b_s, s_l)
#         token_output = model_output["token_output"]  # (s_l, b_s, vocab_size)

#         assert (target.size()[0] == token_output.size()[1])

#         batch_size, seq_len = tuple(target.size())

#         token_pred = token_output.reshape((seq_len * batch_size, -1)).float()
#         loss = self.loss_fn(
#             token_pred, target.reshape(-1)
#         ).reshape((seq_len, batch_size))

#         inv_target_mask = ~(target_mask > 0)
#         num_tokens = inv_target_mask.sum()
#         loss = loss.sum() / num_tokens

#         return loss

#     def _calc_perplexity(self, batch_input, model_output):
#         target_ids = batch_input["target"]  # bs, seq_len
#         target_mask = batch_input["target_mask"]  # bs, seq_len
#         vocab_dist_output = model_output["token_output"]  # seq_len, bs

#         inv_target_mask = ~(target_mask > 0)

#         # choose probabilities of token indices
#         # logits = log_probabilities
#         log_probs = vocab_dist_output.transpose(
#             0, 1).gather(2, target_ids.unsqueeze(2)).squeeze(2)
#         log_probs = log_probs * inv_target_mask
#         log_probs = log_probs.sum(dim=1)

#         seq_lengths = inv_target_mask.sum(dim=1)
#         exp = - (1 / seq_lengths)
#         perp = torch.pow(log_probs.exp(), exp)
#         return perp.mean()

#     def _calc_my_perplexity(self, batch_input, model_output):
#         target_ids = batch_input["target"]  # bs, seq_len
#         target_mask = batch_input["target_mask"]  # bs, seq_len
#         # seq_len, bs, vocab_size
#         vocab_dist_output = model_output["token_output"]

#         inv_target_mask = ~(target_mask > 0)  # bs, seq_len

#         # seq_len, bs, vocab_size
#         l_probs = F.log_softmax(vocab_dist_output, dim=2)
#         target_l_probs = l_probs.transpose(
#             0, 1).gather(2, target_ids.unsqueeze(2)).squeeze(2)  # bs, seq_len
#         target_l_probs = target_l_probs * inv_target_mask
#         target_l_probs = target_l_probs.sum(dim=1)

#         seq_lengths = inv_target_mask.sum(dim=1)
#         neg_normalized_l_probs = -target_l_probs / seq_lengths
#         perplexity = torch.pow(2, neg_normalized_l_probs)

#         return perplexity.mean(), neg_normalized_l_probs.mean()

#     def _predicted_accuracy(self, batch_input, model_output):
#         target_ids = batch_input["target"]  # bs, seq_len
#         target_mask = batch_input["target_mask"]  # bs, seq_len
#         inv_mask = ~target_mask
#         # seq_len, bs, vocab_size
#         predicted_logits = model_output["token_output"]

#         predicted_ids = torch.argmax(
#             predicted_logits, dim=2).transpose(0, 1)  # bs, seq_len

#         masked_correct = (predicted_ids == target_ids) & inv_mask
#         return torch.sum(masked_correct) / torch.sum(inv_mask)

#     def _full_accuracy(self, batch_input, model_output):
#         target_ids = batch_input["target"]  # bs, seq_len
#         target_mask = batch_input["target_mask"]  # bs, seq_len
#         inv_mask = ~target_mask
#         # seq_len, bs, vocab_size
#         predicted_logits = model_output["token_output"]

#         predicted_ids = torch.argmax(
#             predicted_logits, dim=2).transpose(0, 1)  # bs, seq_len

#         masked_correct = (predicted_ids == target_ids) & inv_mask
#         seq_sum_eq_mask_sum = torch.sum(
#             masked_correct, dim=1) == torch.sum(inv_mask, dim=1)
#         return seq_sum_eq_mask_sum.float().mean()

#     def training_step(self, batch, batch_idx):
#         hsqc, collated_smiles = batch

#         out = self.forward(batch)
#         loss = self._calc_loss(collated_smiles, out)
#         with torch.no_grad():
#             perplexity = self._calc_perplexity(collated_smiles, out)
#             my_perplexity, my_nnll = self._calc_my_perplexity(
#                 collated_smiles, out)
#             accuracy = self._predicted_accuracy(collated_smiles, out)
#             full_accuracy = self._full_accuracy(collated_smiles, out)
#         self.log("tr/loss", loss)
#         metrics = {
#             "loss": loss.detach().item(),
#             "perplexity": perplexity.detach().item(),
#             "my_perplexity": my_perplexity.detach().item(),
#             "my_nnll": my_nnll.detach().item(),
#             "accuracy": accuracy.detach().item(),
#             "full_accuracy": full_accuracy.detach().item()
#         }
#         self.training_step_outputs.append(metrics)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         _, collated_smiles = batch

#         out = self.forward(batch)
#         loss = self._calc_loss(collated_smiles, out)
#         perplexity = self._calc_perplexity(collated_smiles, out)
#         my_perplexity, my_nnll = self._calc_my_perplexity(collated_smiles, out)
#         accuracy = self._predicted_accuracy(collated_smiles, out)
#         full_accuracy = self._full_accuracy(collated_smiles, out)
#         metrics = {
#             "loss": loss.detach().item(),
#             "perplexity": perplexity.detach().item(),
#             "my_perplexity": my_perplexity.detach().item(),
#             "my_nnll": my_nnll.detach().item(),
#             "accuracy": accuracy.detach().item(),
#             "full_accuracy": full_accuracy.detach().item()
#         }
#         self.validation_step_outputs.append(metrics)
#         return metrics

#     @staticmethod
#     def add_model_specific_args(parent_parser, model_name=""):
#         HsqcRankedTransformer.add_model_specific_args(
#             parent_parser, model_name)
#         return parent_parser

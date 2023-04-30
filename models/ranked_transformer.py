import logging
import pytorch_lightning as pl
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from utils import ranker, constants
from models import compute_metrics
from models.encoders.encoder_factory import build_encoder
from models.extras.transformer_stuff import (
    generate_square_subsequent_mask
)
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
import os

from models.chemformer.molbart_utils import (
    PreNormDecoderLayer
)

from utils.lr_scheduler import NoamOpt

RANKED_TNSFMER_ARGS = [
    "dim_model", "dim_coords", "heads", "layers", "ff_dim", "coord_enc", "wavelength_bounds",
    # exclude save_params
    "gce_resolution", "r_dropout", "out_dim", "ranking_set_path",
    "lr", "scheduler", "freeze_weights"
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
      coord_enc="ce",
      wavelength_bounds=None,
      gce_resolution=1,
      r_dropout=0,
      out_dim=6144,
      # other business logic stuff
      save_params=True,
      ranking_set_path="",
      # training args
      lr=1e-3,
      scheduler=None,  # None, "attention"
      freeze_weights=False,
      *args,
      **kwargs,
  ):
    print("\n\n\nRanked Transformer callin super")
    super().__init__()
    params = locals().copy()
    self.out_logger = logging.getLogger("lightning")
    self.out_logger.info("[RankedTransformer] Started Initializing")

    # logging print
    for k, v in params.items():
      if k in RANKED_TNSFMER_ARGS:
        self.out_logger.info(f"[RankedTransformer] {k=},{v=}")

    # don't set ranking set if you just want to treat it as a module
    if ranking_set_path:
      assert (os.path.exists(ranking_set_path))
      self.ranker = ranker.RankingSet(file_path=ranking_set_path)

    if save_params:
      self.save_hyperparameters(*RANKED_TNSFMER_ARGS)

    # ranked encoder
    self.enc = build_encoder(
        coord_enc, dim_model, dim_coords, wavelength_bounds, gce_resolution)
    self.out_logger.info(f"[RankedTransformer] Using {str(self.enc.__class__)}")

    self.loss = nn.BCEWithLogitsLoss()
    self.lr = lr

    self.scheduler = scheduler
    self.dim_model = dim_model

    self.validation_step_outputs = []
    self.training_step_outputs = []

    # === All Parameters ===
    self.fc = nn.Linear(dim_model, out_dim)
    # (1, 1, dim_model)
    self.latent = torch.nn.Parameter(torch.randn(1, 1, dim_model))

    # The Transformer layers:
    layer = torch.nn.TransformerEncoderLayer(
        d_model=dim_model,
        nhead=heads,
        dim_feedforward=ff_dim,
        batch_first=True,
        dropout=r_dropout,
    )
    self.transformer_encoder = torch.nn.TransformerEncoder(
        layer,
        num_layers=layers,
    )
    # === END Parameters ===

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
    parser.add_argument(f"--{model_name}dim_model", type=int, default=128)
    parser.add_argument(f"--{model_name}dim_coords",
                        type=int, default=[43, 43, 42],
                        nargs="*", action="store")
    parser.add_argument(f"--{model_name}heads", type=int, default=8)
    parser.add_argument(f"--{model_name}layers", type=int, default=8)
    parser.add_argument(f"--{model_name}ff_dim", type=int, default=512)
    parser.add_argument(f"--{model_name}wavelength_bounds",
                        type=float, default=None, nargs='+', action='append')
    parser.add_argument(f"--{model_name}dropout", type=float, default=0)
    parser.add_argument(f"--{model_name}out_dim", type=int, default=6144)
    parser.add_argument(f"--{model_name}pos_weight", type=float, default=1.0)
    parser.add_argument(f"--{model_name}weight_decay", type=float, default=0.0)
    parser.add_argument(f"--{model_name}scheduler", type=str, default=None)
    parser.add_argument(f"--{model_name}coord_enc", type=str, default="ce")
    parser.add_argument(f"--{model_name}gce_resolution", type=float, default=1)
    parser.add_argument(f"--{model_name}freeze_weights",
                        type=bool, default=False)
    parser.add_argument(f"--{model_name}ranking_set_path", type=str, default="")
    return parent_parser

  @staticmethod
  def prune_args(vals: dict, model_name=""):
    items = [(k[len(model_name) + 1:], v)
             for k, v in vals.items() if k.startswith(model_name)]
    return dict(items)

  def encode(self, hsqc, mask=None):
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

    points = self.enc(hsqc)
    # Add the spectrum representation to each input:
    latent = self.latent.expand(points.shape[0], -1, -1)
    points = torch.cat([latent, points], dim=1).to(self.device)
    out = self.transformer_encoder(points, src_key_padding_mask=mask)
    return out, mask

  def forward(self, hsqc):
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
    out, _ = self.encode(hsqc)
    out = self.fc(out[:, :1, :].squeeze(1))  # extracts cls token
    return out

  def training_step(self, batch, batch_idx):
    x, labels = batch
    labels = labels.type(torch.float32)
    out = self.forward(x)
    loss = self.loss(out, labels)

    self.log("tr/loss", loss)
    return loss

  def validation_step(self, batch, batch_idx):
    x, labels = batch
    labels = labels.type(torch.float32)
    out = self.forward(x)
    loss = self.loss(out, labels)
    metrics = compute_metrics.cm(
        out, labels, self.ranker, loss, self.loss, thresh=0.0)
    self.validation_step_outputs.append(metrics)
    return metrics

  def on_train_epoch_end(self):
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
    feats = self.validation_step_outputs[0].keys()
    di = {}
    for feat in feats:
      di[f"val/mean_{feat}"] = np.mean([v[feat]
                                       for v in self.validation_step_outputs])
    for k, v in di.items():
      self.log(k, v, on_epoch=True)
    self.validation_step_outputs.clear()

  def configure_optimizers(self):
    if not self.scheduler:
      return torch.optim.Adam(self.parameters(), lr=self.lr)
    elif self.scheduler == "attention":
      optim = torch.optim.Adam(self.parameters(), lr=0, betas=(
          0.9, 0.98), eps=1e-9)
      scheduler = NoamOpt(self.dim_model, 4000, optim)
      return {
          "optimizer": optim,
          "lr_scheduler": {
              "scheduler": scheduler,
              "interval": "step",
              "frequency": 1,
          }
      }


MOONSHOT_ARGS = [
    "pad_token_idx", "vocab_size", "d_model", "num_layers", "num_heads", "d_feedforward",
    "activation", "max_seq_len", "dropout"
]

class Moonshot(HsqcRankedTransformer):
  """
    Only parameters, no sampling
  """

  def __init__(self,
               pad_token_idx,
               vocab_size,
               d_model,
               num_layers,
               num_heads,
               d_feedforward,
               # lr,
               # weight_decay,
               activation,
               # num_steps,
               max_seq_len,
               # schedule,
               # warm_up_steps,
               dropout=0.1,
               *args,
               **kwargs):
    super().__init__(*args, **kwargs)

    # _AbsTransformerModel
    self.pad_token_idx = pad_token_idx
    self.vocab_size = vocab_size
    self.d_model = d_model
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.d_feedforward = d_feedforward
    self.activation = activation
    self.dropout = nn.Dropout(dropout)
    self.max_seq_len = max_seq_len

    # BART stuff
    self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_idx)
    dec_norm = nn.LayerNorm(d_model)
    dec_layer = PreNormDecoderLayer(
        d_model, num_heads, d_feedforward, dropout, activation)
    self.decoder = nn.TransformerDecoder(dec_layer, num_layers, norm=dec_norm)
    self.token_fc = nn.Linear(d_model, vocab_size)
    self.loss_fn = nn.CrossEntropyLoss(
        reduction="none", ignore_index=pad_token_idx)
    self.log_softmax = nn.LogSoftmax(dim=2)
    self.register_buffer("pos_emb", self._positional_embs())

    self.save_hyperparameters(*MOONSHOT_ARGS)

  # Ripped from chemformer
  def _construct_input(self, token_ids):
    """
      Expects tokens in (seq_len, b_s) format

    Returns:
      (seq_len, b_s, d_model) embedding (with dropout applied)
    """
    seq_len, _ = tuple(token_ids.size())
    token_embs = self.emb(token_ids)

    # Scaling the embeddings like this is done in other transformer libraries
    token_embs = token_embs * math.sqrt(self.d_model)

    positional_embs = self.pos_emb[:seq_len, :].unsqueeze(0).transpose(0, 1)
    embs = token_embs + positional_embs
    embs = self.dropout(embs)
    return embs

  # Ripped from chemformer
  def _positional_embs(self):
    """ Produces a tensor of positional embeddings for the model

    Returns a tensor of shape (self.max_seq_len, self.d_model) filled with positional embeddings,
    which are created from sine and cosine waves of varying wavelength
    """

    encs = torch.tensor(
        [dim / self.d_model for dim in range(0, self.d_model, 2)])
    encs = 10000 ** encs
    encs = [(torch.sin(pos / encs), torch.cos(pos / encs))
            for pos in range(self.max_seq_len)]
    encs = [torch.stack(enc, dim=1).flatten()[:self.d_model] for enc in encs]
    encs = torch.stack(encs)
    return encs

  def decode(self, memory, encoder_mask, decoder_inputs, decoder_mask):
    """

    Args:
        memory: (b_s, seq_len, dim_model)
        encoder_padding_mask : (b_s, seq_len)
        decoder_inputs: (seq_len, b_s)
        decoder_mask: (b_s, seq_len)

    Returns:
        {
          model_output: (s_l, b_s, d_model)
          token_output: (s_l, b_s, vocab_size)
        }
    """
    _, s_l = decoder_mask.size()

    # (s_l, s_l)
    tgt_mask = generate_square_subsequent_mask(s_l, device=self.device)
    # (b_s, s_l, dim)
    decoder_embs = self._construct_input(decoder_inputs)

    # embs, memory need seq_len, batch_size convention
    # (s_l, b_s, dim), (s_l, b_s, dim)
    decoder_embs, memory = decoder_embs.transpose(0, 1), memory.transpose(0, 1)

    model_output = self.decoder(
        decoder_embs,
        memory,
        tgt_mask=tgt_mask,  # prevent cheating mask
        tgt_key_padding_mask=decoder_mask,  # padding mask
        memory_key_padding_mask=encoder_mask  # padding mask
    )

    token_output = self.token_fc(model_output)

    return {
        "model_output": model_output,
        "token_output": token_output,
    }

  def forward(self, batch):
    # I use (batch_size, seq_len convention)
    # see datasets/dataset_utils.py:tokenise_and_mask
    hsqc, collated_smiles = batch

    # encoder
    # (b_s, seq_len, dim_model), (b_s, seq_len)
    memory, encoder_mask = self.encode(hsqc)

    # decode
    decoder_inputs = collated_smiles["decoder_inputs"]
    decoder_mask = collated_smiles["decoder_mask"]

    return self.decode(memory, encoder_mask, decoder_inputs, decoder_mask)

  def _calc_loss(self, batch_input, model_output):
    """ Calculate the loss for the model

    Args:
        batch_input (dict): Input given to model,
        model_output (dict): Output from model

    Returns:
        loss (singleton tensor),
    """

    target = batch_input["target"]  # (b_s, s_l)
    target_mask = batch_input["target_mask"]  # (b_s, s_l)
    token_output = model_output["token_output"]  # (s_l, b_s, vocab_size)

    assert (target.size()[0] == token_output.size()[1])

    batch_size, seq_len = tuple(target.size())

    token_pred = token_output.reshape((seq_len * batch_size, -1)).float()
    loss = self.loss_fn(
        token_pred, target.reshape(-1)
    ).reshape((seq_len, batch_size))

    inv_target_mask = ~(target_mask > 0)
    num_tokens = inv_target_mask.sum()
    loss = loss.sum() / num_tokens

    return loss

  def _calc_perplexity(self, batch_input, model_output):
    target_ids = batch_input["target"]  # bs, seq_len
    target_mask = batch_input["target_mask"]  # bs, seq_len
    vocab_dist_output = model_output["token_output"]  # seq_len, bs

    inv_target_mask = ~(target_mask > 0)

    # choose probabilities of token indices
    # logits = log_probabilities
    log_probs = vocab_dist_output.transpose(
        0, 1).gather(2, target_ids.unsqueeze(2)).squeeze(2)
    log_probs = log_probs * inv_target_mask
    log_probs = log_probs.sum(dim=1)

    seq_lengths = inv_target_mask.sum(dim=1)
    exp = - (1 / seq_lengths)
    perp = torch.pow(log_probs.exp(), exp)
    return perp.mean()

  def _calc_my_perplexity(self, batch_input, model_output):
    target_ids = batch_input["target"]  # bs, seq_len
    target_mask = batch_input["target_mask"]  # bs, seq_len
    vocab_dist_output = model_output["token_output"]  # seq_len, bs, vocab_size

    inv_target_mask = ~(target_mask > 0)  # bs, seq_len

    l_probs = F.log_softmax(vocab_dist_output, dim=2)  # seq_len, bs, vocab_size
    target_l_probs = l_probs.transpose(
        0, 1).gather(2, target_ids.unsqueeze(2)).squeeze(2)  # bs, seq_len
    target_l_probs = target_l_probs * inv_target_mask
    target_l_probs = target_l_probs.sum(dim=1)

    seq_lengths = inv_target_mask.sum(dim=1)
    neg_normalized_l_probs = -target_l_probs / seq_lengths
    perplexity = torch.pow(2, neg_normalized_l_probs)

    return perplexity.mean(), neg_normalized_l_probs.mean()

  def _predicted_accuracy(self, batch_input, model_output):
    target_ids = batch_input["target"]  # bs, seq_len
    target_mask = batch_input["target_mask"]  # bs, seq_len
    inv_mask = ~target_mask
    predicted_logits = model_output["token_output"]  # seq_len, bs, vocab_size

    predicted_ids = torch.argmax(
        predicted_logits, dim=2).transpose(0, 1)  # bs, seq_len

    masked_correct = (predicted_ids == target_ids) & inv_mask
    return torch.sum(masked_correct) / torch.sum(inv_mask)

  def training_step(self, batch, batch_idx):
    _, collated_smiles = batch

    out = self.forward(batch)
    loss = self._calc_loss(collated_smiles, out)
    with torch.no_grad():
      perplexity = self._calc_perplexity(collated_smiles, out)
      my_perplexity, my_nnll = self._calc_my_perplexity(collated_smiles, out)
      accuracy = self._predicted_accuracy(collated_smiles, out)

    self.log("tr/loss", loss)
    metrics = {
        "loss": loss.detach().item(),
        "perplexity": perplexity.detach().item(),
        "my_perplexity": my_perplexity.detach().item(),
        "my_nnll": my_nnll.detach().item(),
        "accuracy": accuracy.detach().item()
    }
    self.training_step_outputs.append(metrics)
    return loss

  def validation_step(self, batch, batch_idx):
    _, collated_smiles = batch

    out = self.forward(batch)
    loss = self._calc_loss(collated_smiles, out)
    perplexity = self._calc_perplexity(collated_smiles, out)
    my_perplexity, my_nnll = self._calc_my_perplexity(collated_smiles, out)
    accuracy = self._predicted_accuracy(collated_smiles, out)
    metrics = {
        "loss": loss.detach().item(),
        "perplexity": perplexity.detach().item(),
        "my_perplexity": my_perplexity.detach().item(),
        "my_nnll": my_nnll.detach().item(),
        "accuracy": accuracy.detach().item()
    }
    self.validation_step_outputs.append(metrics)
    return metrics

  @staticmethod
  def add_model_specific_args(parent_parser, model_name=""):
    HsqcRankedTransformer.add_model_specific_args(parent_parser, model_name)
    return parent_parser

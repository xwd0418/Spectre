from pydantic import BaseModel, Field
from typing import Dict, Any, Literal, Optional

class BaseConfig(BaseModel):
    model_name: str
    experiment_name: str
    model_type: Literal["HSQCRankedTransformer", "HSQCRankedResNet"]
    mode: Literal["train", "val", "test", "test_deepsat"] = "train"
    debug: bool = False
    
    log_folder: str
    data_folder: str
    
    epochs: int = 300
    batch_size: int = 32
    accumulate_grad_batches_num: int = 4
    num_workers: int = 4
    
    metric: str = "val/mean_cos"
    metricmode: str = "max"
    patience: int = 7
    
    load_all_weights: str = ""
    freeze: bool = False
    
    checkpoint_path: Optional[str]
    delete_checkpoint: bool = False
    
    fp_choice: Literal["hash_entropy", "db_specific_fp", "mfp_specific_radius"]
    fp_radius: Optional[int]
    normalize_hsqc: bool = False
    disable_solvent: bool = False
    enable_hsqc_delimiter_only_2d: bool = False
    use_peak_values: bool = False
    use_1d_nmr_no_solvent: bool = True
    rank_by_soft_output: bool = True
    use_mw: bool = True
    similarity_measure: Literal["cosine", "jaccard"] = "cosine"
    jittering: float = 0.0
    rank_by_test_set: bool = False
    
    num_class: int = 25
    loss_func: Literal["MSE", "CE"] = "MSE"
    
    input_types: list[Literal["mw", "c", "h", "2d"]] = ["2d"]
    separate_classifier: bool = False
    weighted_sample_based_on_input_type: bool = False
    weighted_sampling_strategy: str = "none"
    random_seed: int = 42
    train_on_all_info_set: bool = False
    
    fp_building_type: Literal["normal", "exact"] = "normal"
    out_dim: int = 6144

class HSQCRankedTransformerConfig(BaseModel):
    lr: float = 1e-5
    noam_factor: float = 1.0
    dim_model: int = 784
    dim_coords: list[int]
    heads: int = 8
    layers: int = 16
    ff_dim: int = 3072
    wavelength_bounds: list[list[float]] = [[0.01, 400.0], [0.01, 20.0]]
    dropout: float = 0.1
    pos_weight: Optional[str | float] = None
    weight_decay: float = 0.0
    l1_decay: float = 0.0
    warmup_steps: int = 8000
    scheduler: str = "attention"
    coord_encoder: str = "sce"
    gce_resolution: float = 1
    freeze_weights: bool = False
    ranking_set_path: str = ""

class HSQCRankedResNetConfig(BaseModel):
    pass

_model_constructor = {
    'HSQCRankedTransformer': HSQCRankedTransformerConfig,
    'HSQCRankedResNet': HSQCRankedResNetConfig
}

class ProjectConfig(BaseModel):
    config: dict
    config_name: str = ""
    base_config: BaseConfig = None
    model_args: HSQCRankedTransformerConfig | HSQCRankedResNetConfig = None
    model_type: str = ""

    def model_post_init(self, __context):
        self.base_config = BaseConfig(**self.config["base_config"])
        self.config_name = self.base_config.experiment_name
        self.model_type = self.base_config.model_type
        if self.model_type == 'HSQCRankedResNet':
            raise NotImplementedError()
        constructor = _model_constructor.get(self.model_type, None)
        if constructor is None:
            raise ValueError('Invalid model type')
        self.model_args = constructor(**self.config["model_args"])

from typing import Union
import math

# Configuration for training
training_config = {
    "batch_size": 64,
    "learning_rate": 0.0001,
    "num_steps": 400000
}



# Configuration for selective copying dataset
copy_dataset_config = {
    "l_noise": 4096,  # number of padding tokens
    "l_memorize": 16,  # number of tokens to memorize
    "n_tokens": 16,  # alphabet size
    "lag": False,
    "variable": True,  # Randomly distribute memorization tokens throughout sequence instead of frontloading them
    "variable_length": False,  # Randomize number of tokens to memorize
    "one_hot": False,
    "reverse": False,
    "static": False,
}

# Configuration for Mamba model
class MambaConfig:
    d_model: int = 64
    n_layer: int = 2
    vocab_size: int = copy_dataset_config['n_tokens']
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int,str] = math.ceil(d_model / 16)
    d_conv: int = 4
    conv_bias:bool = True
    bias: bool = False
    scan_mode: str = 'cumsum'
    d_inner = expand * d_model

    ssm_cfg: dict = {}
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 1
    tie_embeddings: bool = False

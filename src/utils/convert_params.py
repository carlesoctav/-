import jax.numpy as jnp
import typing as tp
from flax.traverse_util import flatten_dict
import torch



def process_single_params(
    key: str,
    tensor: torch.Tensor, 
    config: tp.Dict[str, tp.Any],
    dtype: tp.Any = jnp.float32
) -> tp.Tuple[tp.Optional[tp.Tuple], tp.Optional[jnp.ndarray]]:

    new_key = key
    if any(layer_name in key for layer_name in config["embedding_layer_names"]):
        new_key = f"{key[:-len('.weight')]}.embedding"

    elif any(layer_norm in key for layer_norm in config["layernorm_names"]):
        new_key = key.replace(".weight", ".scale")

    # Handle regular weights
    elif "weight" in key:
        ndim = len(tensor.shape)
        match ndim:
            case 2:
                tensor = tensor.transpose(0, 1)
            case 3:
                tensor = tensor.transpose(0, 2)
            case 4:
                # 2d conv layers
                tensor = tensor.permute(2, 3, 1, 0)
            case 5:
                # 3d conv layers
                tensor = tensor.permute(2, 3, 4, 1, 0)
            case 6:
                # 4d conv layers
                tensor = tensor.permute(4, 5, 3, 2, 1, 0)
            case _:
                ...
        new_key = key.replace(".weight", ".kernel")

    key_tuple = tuple(int(n) if n.isdigit() else n for n in new_key.split("."))

    # Skip if using tied embeddings and this is the language model head
    if config["uses_tie_word_embedding"] and config["lm_head_name"]:
        if key_tuple[0] == config["lm_head_name"]:
            return None, None

    if "bfloat16" in str(tensor.dtype):
        tensor = tensor.float()

    return key_tuple, jnp.asarray(tensor.cpu().detach().numpy(), dtype=dtype)

def convert_dict_state_to_flax_params(
    state_dict: tp.Dict[str, tp.Any],
    embedding_layer_names: tp.List[str],
    layernorm_names: tp.List[str]
):
    config = {
        "embedding_layer_names": embedding_layer_names,
        "layernorm_names": layernorm_names,
        "uses_tie_word_embedding": False,
        "lm_head_name": None
    }
    flatten_params = {}
    for key, tensor in state_dict.items():
        key_tuple, tensor = process_single_params(key, tensor, config)
        if key_tuple is not None and tensor is not None:
            flatten_params[key_tuple] = tensor

    return flatten_params


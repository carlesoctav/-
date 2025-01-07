import jax
import jax.numpy as jnp
import typing as tp
import torch
from tqdm.auto import tqdm

from flax import nnx

from flax.typing import PathParts


K = tp.TypeVar("K", nnx.Module, nnx.Variable)


def iter_module(
    module: nnx.Module,
    module_type: tp.List[tp.Type[K]] = [nnx.Module],
) -> tp.Generator[tp.Tuple[PathParts, K], None, None]:
    for name, value in nnx.iter_graph(module):
        if any(isinstance(value, t) for t in module_type):
            yield name, value


def finding_embedding_names(module: nnx.Module) -> tp.List[str]:
    return list(set(
        str(path[-1]) for path, value in iter_module(module, module_type=[nnx.Embed])
    ))


def finding_layernorm_names(module: nnx.Module) -> tp.List[str]:
    return list(set(
        str(path[-1]) for path, value in iter_module(module, module_type=[nnx.LayerNorm])
    ))

def process_single_state(
    key: str,
    tensor: torch.Tensor,
    config: tp.Dict[str, tp.Any],
) -> tp.Tuple[tp.Tuple, jnp.ndarray]:

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

    if "bfloat16" in str(tensor.dtype):
        tensor = tensor.float()

    return key_tuple, jnp.asarray(tensor.cpu().detach().numpy(), dtype=config["param_dtype"])


def convert_torch_state_to_flax_state(
    state_dict: tp.Dict[str, tp.Any],
    lazy_model: nnx.Module,
    param_dtype: jnp.dtype = jnp.float32,
):
    embedding_layer_names = finding_embedding_names(lazy_model)
    layernorm_names = finding_layernorm_names(lazy_model)

    state, others = nnx.state(lazy_model, nnx.Param, ...)

    print(f"DEBUGPRINT[5]: paramaters_transformations.py:85: others={others}")
    jax.tree_util.tree_map_with_path(lambda x, y: print(x, y), others)

    config = {
        "embedding_layer_names": embedding_layer_names,
        "layernorm_names": layernorm_names,
        "param_dtype": param_dtype,
    }
    flatten_params = {}
    with tqdm(total=len(state_dict)) as pbar:
        for key, tensor in state_dict.items():
            key_tuple, tensor = process_single_state(key, tensor, config)
            flatten_params[key_tuple] = tensor
            pbar.update(1)
            del tensor
    return flatten_params


def recreate_rngs_state_for_lazy_model(others):

    def f(x):
        pass
    return jax.tree_util.tree_map(lambda x: x, others, is_leaf = lambda x: isinstance(x, nnx.VariableState)) 

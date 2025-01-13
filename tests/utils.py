import jax.numpy as jnp
from flax.nnx import traversals
import torch
from flax import nnx

from src.utils.paramaters_transformations import convert_torch_state_to_flax_state, recreate_rngs_state_for_lazy_model

def move_model_weight(model: nnx.Module, torch_model: torch.nn.Module):
    state_dict = torch_model.state_dict()

    flatten_params = convert_torch_state_to_flax_state(state_dict, model)
    unflatten_params = traversals.unflatten_mapping(flatten_params)

    params, others = nnx.state(model, nnx.Param, ...)
    params.replace_by_pure_dict(unflatten_params)
    others = recreate_rngs_state_for_lazy_model(others)
    nnx.update(model, params, others)


def to_numpy(x):
    return x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x


def assert_close(right, left, n_diffs=5, atol = 0.125, rtol = 0):
    right = to_numpy(right) 
    left = to_numpy(left)
    diff_mask =  ~jnp.isclose(right, left, atol=atol, rtol=rtol)
    diff_indices = jnp.where(diff_mask.reshape(-1))[0]
    nfails = jnp.sum(diff_mask)

    if nfails > 0:
        print(f"Number of fails: {nfails}")
        for  _, idx in enumerate(diff_indices[:n_diffs]):
            print(f"Index: {idx}, Right: {right.reshape(-1)[idx]}, Left: {left.reshape(-1)[idx]}")
        raise AssertionError(f"Number of fails: {nfails}")

    print(f"Last 5 comparison: {right.reshape(-1)[-5:]}, {left.reshape(-1)[-5:]}")

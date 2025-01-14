import jax
from flax import nnx
import typing as tp
from jax.sharding import PartitionSpec
import warnings
import re


A = tp.TypeVar('A')


def match_partition_spec(
    tree: A,
    rules: tp.Tuple[str, PartitionSpec] 
)-> A: 
    def _maybe_replicate(x):
        if hasattr(x, 'shape'):
            return PartitionSpec()
        else:
            return None

    def get_partition_spec(name, leaf):
        if isinstance(leaf, nnx.VariableState):
            for rule, ps in rules:
                if re.search(rule, name) is not None:
                    if len(ps) > leaf.value.ndim:
                        ps = PartitionSpec(*tuple(ps[: leaf.ndim]))
                        warnings.warn(
                            f"PartitionSpec Related to {name} went out of range (will be auto trimed to {ps}).",
                            stacklevel=1,
                        )
                    return leaf.replace(ps)
            
            else:
                leaf.replace(_maybe_replicate(leaf.value))
        else:
            return leaf.replace(_maybe_replicate(leaf.value)) 



    return jax.tree_util.tree_map_with_path(
        lambda name, leaf: get_partition_spec(tree_path_to_string(name, sep="/"), leaf),
        tree,
        is_leaf=lambda x: isinstance(x, nnx.VariableState),
    )


def tree_path_to_string(path: tp.Tuple, sep: tp.Optional[str] = None) -> str | tp.Tuple:
    keys = []
    for key in path:
        if isinstance(key, jax.tree_util.SequenceKey):
            keys.append(str(key.idx))
        elif isinstance(key, jax.tree_util.DictKey):
            keys.append(str(key.key))
        elif isinstance(key, jax.tree_util.GetAttrKey):
            keys.append(str(key.name))
        elif isinstance(key, jax.tree_util.FlattenedIndexKey):
            keys.append(str(key.key))
        else:
            keys.append(str(key))
    if sep is None:
        return tuple(keys)  # Return a tuple of strings if no separator
    return sep.join(keys)



def params_sharder(model: nnx.Module, partition_spec, mesh):
    params = nnx.state(model)
    sharded_state = nnx.with_sharding_constraint(params, partition_spec, mesh)
    nnx.update(model, sharded_state)
    return model


def is_sharded(model: nnx.Module) -> bool:
    # another todo
    return False


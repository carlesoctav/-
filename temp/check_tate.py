from flax import nnx
from torch import nn
from jax.sharding import PartitionSpec

default_kernel_init = nnx.initializers.lecun_normal()

class Model(nnx.Module):

    def __init__(self, rngs):

        self.linear = nnx.Linear(10,10, rngs = rngs, kernel_init=nnx.with_partitioning(default_kernel_init, sharding = (None,)))
        self.dropout = nnx.Dropout(0.3, rngs = rngs)
        self.layernorm = nnx.BatchNorm(num_features=10, rngs = rngs)



a = Model(nnx.Rngs(0))
state = nnx.state(a)
a = nnx.get_partition_spec(state)
print(f"DEBUGPRINT[3]: check_tate.py:16: a={a}")

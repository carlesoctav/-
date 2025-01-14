import functools
import jax
import typing as tp
from flax import nnx
from jax.sharding import PartitionSpec
import optax

import warnings
from configuration_utils import ParallelConfig, PartitionTuple
from src.distributed.mesh_utils import initialize_mesh
from src.distributed.sharding import is_sharded, match_partition_spec, params_sharder


class TrainerUtil:
    def __init__(
        self,
        #TODO: fix this type later 
        mode: tp.Literal["dp", "ddp", "custom"],
        num_devices: tp.Optional[int]  = None,
        multihost:  bool =  False
    ):
        self.mode = mode
        if num_devices == None:
            self.num_devices = len(jax.devices())
        else:
            if num_devices > len(jax.devices()):
                warnings.warn(
                    f"Number of devices requested is greater than the number of available devices. Using {len(jax.devices())} devices instead."
                )
                self.num_devices = len(jax.devices())
            else:
                self.num_devices = num_devices

    def loss_function(self):
        raise NotImplementedError

    def train_step(
        self,
    ):
        pass

    def eval_step(
        self,
    ):
        pass

    def setup(
        self,
        model: nnx.Module,
        optax_optimizer: optax.GradientTransformation, 
    )-> tp.Tuple[nnx.Module, nnx.Optimizer]:

        if is_sharded(model):
            warnings.warn(
                "Model is sharded, before calling setup, for any mode other than 'custom', this sharding will be removed." 
            )

        model= self.shard_model(model)
        optimizer = nnx.Optimizer(model, optax_optimizer)
        return model, optimizer 


    def shard_model(self, model: nnx.Module):
        if self.mode == "dp" or self.mode == "ddp":
            parralel_config = ParallelConfig(partition_tuple = PartitionTuple())
            mesh = initialize_mesh(parralel_config)
            partition_spec = match_partition_spec(nnx.state(model), (".*", PartitionSpec()))
            params_sharder_jitted = nnx.jit(functools.partial(params_sharder, partition_spec=partition_spec, mesh=mesh))
            return params_sharder_jitted(model)
        else:
            return model


    def setup_dataloader(self, dataloader):
        pass

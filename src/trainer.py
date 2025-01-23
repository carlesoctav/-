from enum import Enum
import functools
import jax
import typing as tp
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec
import optax
from torch.utils.data import DataLoader

import warnings
from src.configuration_utils import ParallelConfig, PartitionTuple
from src.distributed.mesh_utils import initialize_mesh
from src.distributed.sharding import is_sharded, match_partition_spec, params_sharder
from src.wrappers import _JaxDataLoader


class ModelShardinEnum(Enum):
    pass

class TrainerUtil:
    def __init__(
        self,
        #TODO: fix this type later 
        parallel_config: ParallelConfig = ParallelConfig(),
        model_sharding: tp.Optional[ModelShardinEnum] = None, 
        multihost:  bool =  False
    ):
        self.multihost = multihost
        self.parallel_config = parallel_config
        self.model_sharding = model_sharding 
        if self.multihost:
            jax.distributed.initialize()

        self.mesh = initialize_mesh(self.parallel_config)

    def loss_function(self, model: nnx.Module, batch):
        raise NotImplementedError

    def setup(
        self,
        model: nnx.Module,
        optax_optimizer: optax.GradientTransformation, 
        debug: bool = False,
    )-> tp.Tuple[nnx.Module, nnx.Optimizer]:

        if is_sharded(model):
            warnings.warn(
                "Model is sharded, before calling setup, for any mode other than 'custom', this sharding will be removed." 
            )

        model= self.shard_model(model)
        optimizer = nnx.Optimizer(model, optax_optimizer)

        def train_step(model, optimizer, batch):
            (loss, others), grads= nnx.value_and_grad(self.loss_function, has_aux = True)(model, batch)
            optimizer.update(grads)
            return (loss, others), grads


        if debug:
            self.train_step = train_step
            self.eval_step = self.loss_function 
        else:
            self.train_step = nnx.jit(train_step)
            self.eval_step = nnx.jit(self.loss_function)

        return model, optimizer 


    def shard_model(self, model: nnx.Module):
        if self.model_sharding == None: 
            partition_rule = ((".*", PartitionSpec()),)
            partition_spec_tree = match_partition_spec(nnx.state(model), partition_rule) 
            print(f"DEBUGPRINT[1]: trainer.py:70: partition_spec_tree={partition_spec_tree}")
            params_sharder_jitted = nnx.jit(functools.partial(params_sharder, partition_spec_tree=partition_spec_tree, mesh=self.mesh))
            return params_sharder_jitted(model)
        else:
            return model


    def setup_dataloader(self, dataloader: DataLoader):
        batch_sharding = NamedSharding(self.mesh, PartitionSpec(("dp")))
        custom_loader = _JaxDataLoader(dataloader=dataloader, sharding=batch_sharding)
        custom_loader = tp.cast(DataLoader, custom_loader)
        return custom_loader


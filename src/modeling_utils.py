from abc import abstractmethod
import functools
import typing as tp
from flax import nnx
from src.distributed.sharding import match_partition_spec, params_sharder
from src.configuration_utils import NNXPretrainedConfig
from transformers import PretrainedConfig
from transformers.utils import (
    logging,
)
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, PartitionSpec


ORBAX_WEIGHTS_NAME = "nnx_model.msgpack"
logger = logging.get_logger(__name__)


def load_sharded_checkpoint(model, folder):
    pass


class NNXPretrainedModel(nnx.Module):
    config_class: tp.Type[PretrainedConfig]
    base_model_prefix: str
    _model_type: str
    _model_task = None
    _missing_keys = set()

    def __init__(
        self,
        config: NNXPretrainedConfig | None,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: tp.Optional[lax.Precision] = None,
        shard_models: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        if not isinstance(config, NNXPretrainedConfig):
            raise TypeError("config must be an instance of NNXPreTrainedConfig")

        self.config = config
        self.name_or_path = config.name_or_path
        _ = self.params_shape
        # _ = sel.model_task
        # _ = self.model_type

    #TODO: think a better approach
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


    @classmethod
    def lazy_init(
        cls: tp.Type["NNXPretrainedModel"], *args, **kwargs
    ) -> "NNXPretrainedModel":
        return nnx.eval_shape(lambda: cls(*args, **kwargs))

    @property
    def params_shape(self):
        """Evaluates the shape of the model's parameters and returns a dictionary."""
        shape_tree = nnx.eval_shape(lambda: nnx.state(self, nnx.Param))
        return shape_tree

    def get_current_model_partition_spec(self):
        pass

    def _get_partition_rules(self, partition_rules):
        """Retrieves the partition rules from input or the config"""
        if partition_rules is None:
            if not hasattr(self, "config"):
                raise ValueError(
                    "Partition rules must be provided either as an argument or through the model config."
                )

            return self.config.get_partition_rules()
        return partition_rules

    def get_partition_spec(self, partition_rules=None):
        partition_rules = self._get_partition_rules(partition_rules)

        return match_partition_spec(self.params_shape, partition_rules)

    def get_mesh(self, mesh: tp.Optional[Mesh] = None) -> Mesh:
        """Retrieves the mesh, either from the provided argument or the config."""
        if mesh is None:
            if (
                not hasattr(self, "config")
                or not hasattr(self.config, "mesh")
                or self.config.mesh is None
            ):
                raise ValueError(
                    "A mesh must be provided, either as an argument or through the model config."
                )
            return self.config.mesh
        return mesh

    @property
    def model_task(self):
        return self.config.model_task

    @property
    def model_type(self):
        return self._model_type

    def shard_model(
        self,
        partition_rules: tp.Optional[tp.Tuple[str, PartitionSpec]] = None,
        mesh: tp.Optional[Mesh] = None,
    ):
        mesh = self.get_mesh(mesh)
        partition_spec = self.get_partition_spec(partition_rules)
        sharder = nnx.jit(
            functools.partial(params_sharder, partition_spec=partition_spec, mesh=mesh)
        )

        self = sharder(self)

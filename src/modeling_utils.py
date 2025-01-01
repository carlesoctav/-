import os
import typing as tp
from flax import nnx
from src.configuration_utils import NNXPreTrainedConfig
from transformers import PretrainedConfig
from transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    cached_file,
    logging,
)


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
        self, config: NNXPreTrainedConfig | None, do_partition: bool = False, **kwargs
    ):
        if not isinstance(config, NNXPreTrainedConfig):
            raise TypeError("config must be an instance of NNXPreTrainedConfig")

        self.config = config
        self.name_or_path = config.name_or_path
        _ = self.mesh
        # _ = self.graphtree_params_shape
        # _ = self.model_task
        # _ = self.model_type

    @classmethod
    def lazy_init(
        cls: tp.Type["NNXPretrainedModel"], *args, **kwargs
    ) -> "NNXPretrainedModel":
        return nnx.eval_shape(lambda: cls(*args, **kwargs))

    @property
    def params_shape(self):
        pass

    def mesh(self):
        self.config.mesh

    @property
    def model_task(self):
        return self.config.model_task

    @property
    def model_type(self):
        return self._model_type

    def shard_model(self):
        pass

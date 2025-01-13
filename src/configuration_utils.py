from abc import abstractmethod
from jax.sharding import Mesh
from jax.experimental.mesh_utils import create_device_mesh
import jax
import transformers
import typing as tp

from dataclasses import dataclass, field

from typing import NamedTuple, Optional, Tuple, Union

from distributed.mesh_utils import initialize_mesh


AxisType = Tuple[int, Union[Tuple[str, ...], str]]


class PartitionTuple(NamedTuple):
    data_axis: AxisType = (-1, "dp")  # TODO: what about (-1, ("dp", "fsdp"))?
    model_axis: AxisType = (1, "tp")
    fsdp_axis: AxisType = (1, "fsdp")
    pp_axis: AxisType = (1, "pp")


@dataclass(kw_only=True, frozen=False)
class ParallelConfig:
    partition_tuple: PartitionTuple = field(default_factory=PartitionTuple)

class NNXPretrainedConfig(transformers.PretrainedConfig):

    def __init__(
        self,
        parallel_config: ParallelConfig = ParallelConfig(),
        **kwargs,
    ):
        self.parallel_config = getattr(self, "parallel_config", parallel_config) 
        super().__init__(**kwargs)

    @property
    def mesh(self):
        return initialize_mesh(self.parallel_config)

    @abstractmethod
    def get_partition_rules():
        raise NotImplementedError

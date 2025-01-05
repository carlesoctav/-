from jax.sharding import Mesh, PartitionSpec as P
import jax.numpy as jnp
from jax.experimental.mesh_utils import create_device_mesh
import jax
import transformers
import typing as tp


from typing import NamedTuple, Optional, Tuple, Union

AxisType = Optional[Union[Tuple[str, ...], str]]


class PartitionAxis(NamedTuple):

    batch_axis: AxisType = ("fsdp", "dp")
    sequence_axis: AxisType = "sp"
    query_sequence_axis: AxisType = "sp"
    head_axis: AxisType = "tp"
    key_sequence_axis: AxisType = "sp"
    hidden_state_axis: AxisType = "tp"
    attention_dim_axis: AxisType = None
    bias_head_sequence_axis: AxisType = None
    bias_key_sequence_axis: AxisType = None

    generation_query_sequence_axis: AxisType = None
    generation_head_axis: AxisType = "tp"
    generation_key_sequence_axis: AxisType = "sp"
    generation_attention_dim_axis: AxisType = None


class NNXPretrainedConfig(transformers.PretrainedConfig):

    def __init__(
        self,
        axis_dims: tp.Sequence[int] = (-1, 1, 1, 1),
        axis_names: tp.Sequence[str] = ("dp", "fsdp", "pp", "tp"),
        partition_axis = PartitionAxis(),
        **kwargs,
    ):
        self.axis_dims = getattr(self, "axis_dims", axis_dims)
        self.axis_names = getattr(self, "axis_names", axis_names)
        self.partition_axis = getattr(self, "partition_axis", partition_axis)
        super().__init__(**kwargs)
    
    @staticmethod
    #TODO
    def create_mesh(
		axis_dims: tp.Sequence[int] = (1, -1, 1, 1),
		axis_names: tp.Sequence[str] = ("dp", "fsdp", "pp", "tp"),
		backend="",
	)-> Mesh:

        devices = jax.devices() if backend == "" else jax.devices(backend)
        shaped_devices = create_device_mesh(
            mesh_shape = axis_dims,
            devices=devices,
        )
        return Mesh(devices = shaped_devices, axis_names = axis_names)


    @property
    def mesh(self):
        return self.create_mesh(
        axis_dims=(
            [v for k, v in self.axis_dims.items()]
            if isinstance(self.axis_dims, dict)
            else self.axis_dims
        ),
        axis_names=(
            [v for k, v in self.axis_names.items()]
            if isinstance(self.axis_names, dict)
            else self.axis_names
        ),
        backend=(
            (self.backend if self.backend is not None else "")
            if hasattr(self, "backend")
            else ""
        ),
        )


    def partition_rules():
        pass

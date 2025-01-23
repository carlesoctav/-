import logging
import os

import jax
import numpy as np
from jax.sharding import Mesh
import typing as tp

if tp.TYPE_CHECKING:
    from src.configuration_utils import ParallelConfig


def initialize_mesh(
    parallel_config: "ParallelConfig", device_array: np.ndarray | None = None
) -> Mesh:

    data_axis_size, data_axis_name = parallel_config.partition_tuple.data_axis
    fsdp_axis_size, fsdp_axis_name = parallel_config.partition_tuple.fsdp_axis
    pipeline_axis_size, pipeline_axis_name = parallel_config.partition_tuple.pp_axis
    model_axis_size, model_axis_name = parallel_config.partition_tuple.model_axis

    if device_array is None:
        device_array = np.array(jax.devices())

    mesh_shape = (
        data_axis_size,
        fsdp_axis_size,
        pipeline_axis_size,
        model_axis_size,
    )

    min_req_devices = np.abs(np.prod(mesh_shape))

    assert device_array.size >= min_req_devices, (
        f"Device array size {device_array.size} is less than minimum required devices {min_req_devices}"
        f" in the requested mesh shape {mesh_shape}."
    )
    device_array = device_array.reshape(mesh_shape)

    mesh = Mesh(
        device_array,
        (
            data_axis_name,
            fsdp_axis_name,
            pipeline_axis_name,
            model_axis_name,
        ),
    )
    if jax.process_index() == 0:
        logging.info(f"Initialized mesh with {mesh}.")

    return mesh


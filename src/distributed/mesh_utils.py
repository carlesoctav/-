#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import logging
import os

import jax
import numpy as np
from jax.sharding import Mesh

from src.configuration_utils import ParallelConfig


def initialize_mesh(
    parallel_config: ParallelConfig, device_array: np.ndarray | None = None, init_distributed_on_slurm: bool = True
) -> Mesh:
    """Initialize the mesh for parallel training.

    Args:
        parallel_config: A dictionary containing the parallelization parameters.
        device_array: A numpy array containing the device structure. If None, all global devices are used.
        init_distributed_on_slurm: Whether to initialize the JAX distributed system, i.e. multiprocess training,
            if SLURM environment variables are present. If False, the JAX distributed system is not initialized.

    Returns:
        The initialized mesh.
    """
    if init_distributed_on_slurm and "SLURM_STEP_NODELIST" in os.environ:
        # Initializes one process per device, using the SLURM environment variables.
        # TODO: We may need to do this already before data loading, so very early in the run script.
        # To be checked once the framework is more mature.
        jax.distributed.initialize()

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


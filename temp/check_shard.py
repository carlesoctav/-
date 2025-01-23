import jax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.sharding import NamedSharding
from jax.experimental import mesh_utils


v1s = jax.random.normal(jax.random.PRNGKey(0), (2, 2))

mesh = Mesh(mesh_utils.create_device_mesh((2, 2)), axis_names=("batch", "features"))
sharding = NamedSharding(mesh, P(None))
v1sp = jax.device_put(v1s, sharding)
jax.debug.visualize_array_sharding(v1sp)
jax.debug.visualize_array_sharding(v1s)

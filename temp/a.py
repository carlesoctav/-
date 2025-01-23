import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

a = jax.random.normal(jax.random.PRNGKey(0), (4, 3))
partition_spec = PartitionSpec("dp", None)
mesh = Mesh(jax.devices(), ("dp",))
sharding = NamedSharding(mesh, partition_spec)

a = jax.device_put(a, NamedSharding(mesh, partition_spec))
print(f"DEBUGPRINT[1]: a.py:9: a={a}")
jax.debug.visualize_array_sharding(a)

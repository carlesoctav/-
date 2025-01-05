import jax
import jax.numpy as jnp

key = jax.random.key(100)
key, q_key, k_key, v_key = jax.random.split(key, 4)
q = jax.random.normal(q_key, (100, 10, 3, 20))
k = jax.random.normal(k_key, (100, 5, 3, 20))
v = jax.random.normal(v_key, (100, 10, 3,20))

print(f"DEBUGPRINT[3]: test_einsum.py:10: q.shape[:-1]={q.shape[:-1]}")

attn = jnp.einsum("...qhd,...khd->qhk", q, k)



new_score = jnp.einsum("...hqk,...vhd->hqd" , attn, v)
print(f"DEBUGPRINT[1]: test_einsum.py:10: attn={attn.shape}")
print(f"DEBUGPRINT[2]: test_einsum.py:13: new_score={new_score.shape}")

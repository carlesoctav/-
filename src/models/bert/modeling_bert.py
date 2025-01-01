from flax import nnx

nnx.dot_product_attention
import jax
import jax.numpy as jnp
import typing as tp
from jax import lax

from src.models.bert.configuration_bert import BertConfig


class BertEmbeddings(nnx.Module):
    def __init__(
        self,
        config: BertConfig,
        dtype: jnp.dtype = jnp.float32,  # the dtype of the computation
        param_dtype: jnp.dtype = jnp.float32,
        precision: tp.Optional[lax.Precision] = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.word_embeddings = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.position_embeddings = nnx.Embed(
            num_embeddings=config.max_position_embeddings,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.token_type_embeddings = nnx.Embed(
            num_embeddings=config.type_vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.LayerNorm = nnx.LayerNorm(
            num_features=config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(rate=config.hidden_dropout_prob, rngs=rngs)
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )

    def __call__(
        self,
        input_ids: jax.Array,
        token_type_ids: jax.Array,
        position_ids: jax.Array,
    ):
        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nnx.Module):
    def __init__(
        self,
        config: BertConfig,
        dtype: jnp.dtype = jnp.float32,  # the dtype of the computation
        param_dtype: jnp.dtype = jnp.float32,
        precision: tp.Optional[lax.Precision] = None,
        *,
        rngs: nnx.Rngs,
    ):
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nnx.Linear(
            in_features=self.hidden_size,
            out_features=self.all_head_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.key = nnx.Linear(
            in_features=self.hidden_size,
            out_features=self.all_head_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.value = nnx.Linear(
            in_features=self.hidden_size,
            out_features=self.all_head_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.dropout = nnx.Dropout(rate=config.attention_probs_dropout_prob, rngs=rngs)

    def _split_head(
        self,
        tensor: jax.Array,
    ):
        new_shape = tensor.shape[:2] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        tensor = tensor.reshape(new_shape)
        return tensor

    def __call__(
        self,
        hidden_states: jax.Array,
        attention_mask: jax.Array | None = None,
    ):
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        q = self._split_head(q)
        k = self._split_head(k)
        v = self._split_head(v)
        attention_output = nnx.dot_product_attention(
            query=q,
            key=k,
            value=v,
            mask=attention_mask,
        )

        return attention_output



class BertIntermediate(nnx.Module):
    def __init__(
        self,
        config: BertConfig,
        dtype: jnp.dtype = jnp.float32,  # the dtype of the computation
        param_dtype: jnp.dtype = jnp.float32,
        precision: tp.Optional[lax.Precision] = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.dense = nnx.Linear(
            config.hidden_size,
            config.intermediate_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.intermediate_act_fn = nnx.gelu

    def __call__(
        self,
        hidden_states: jax.Array,
    ) -> jax.Array:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertSelfOutput(nnx.Module):
    def __init__(
        self,
        config,
        dtype: jnp.dtype = jnp.float32,  # the dtype of the computation
        param_dtype: jnp.dtype = jnp.float32,
        precision: tp.Optional[lax.Precision] = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.dense = nnx.Linear(
            config.hidden_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.LayerNorm = nnx.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(config.hidden_dropout_prob, rngs=rngs)

    def __call__(
        self,
        hidden_states: jax.Array, 
        input_tensor: jax.Array, 
    ) -> jax.Array: 
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



class BertLayer(nnx.Module):
    def __init__(
        self,
        config: BertConfig,
        dtype: jnp.dtype = jnp.float32,  # the dtype of the computation
        param_dtype: jnp.dtype = jnp.float32,
        precision: tp.Optional[lax.Precision] = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.attention = BertSelfAttention(config, dtype, param_dtype, precision, rngs=rngs)
        self.intermediate = BertIntermediate(config, dtype, param_dtype, precision, rngs=rngs)
        self.output = BertSelfOutput(config, dtype, param_dtype, precision, rngs=rngs)

    def __call__(
        self,
        hidden_states: jax.Array, 
        attention_mask: jax.Array | None = None, 
    ) -> jax.Array:
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, hidden_states)
        return layer_output


class BertEncoder(nnx.Module):
    def __init__(
        self,
        config: BertConfig,
        dtype: jnp.dtype = jnp.float32,  # the dtype of the computation
        param_dtype: jnp.dtype = jnp.float32,
        precision: tp.Optional[lax.Precision] = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.layers = [BertLayer(config, dtype, param_dtype, precision, rngs=rngs) for _ in range(config.num_hidden_layers)]

    def forward(
        self,
        hidden_states: jax.Array, 
        attention_mask: jax.Array | None = None, 
    ) -> jax.Array:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states


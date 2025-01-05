from flax import nnx

import jax
import jax.numpy as jnp
import typing as tp
from jax import lax
import transformers

from src.models.bert.configuration_bert import BertConfig
from src.modeling_utils import NNXPretrainedModel
from src.mixins import HuggingFaceCompatible


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

    def _merge_head(
        self,
        tensor: jax.Array,
    ):
        new_shape = tensor.shape[:2] + (self.all_head_size,)
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

        attention_output = self._merge_head(attention_output)

        return attention_output


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


class BertAttention(nnx.Module):
    def __init__(
        self,
        config: BertConfig,
        dtype: jnp.dtype = jnp.float32,  # the dtype of the computation
        param_dtype: jnp.dtype = jnp.float32,
        precision: tp.Optional[lax.Precision] = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.self = BertSelfAttention(
            config, dtype, param_dtype, precision, rngs=rngs
        )
        self.output = BertSelfOutput(config, dtype, param_dtype, precision, rngs=rngs)

    def __call__(
        self,
        hidden_states: jax.Array,
        attention_mask: jax.Array | None = None,
    ) -> jax.Array:
        attention_output = self.self(hidden_states, attention_mask)
        layer_output = self.output(attention_output, hidden_states)
        return layer_output


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


class BertOutput(nnx.Module):
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
            config.intermediate_size,
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
        self.attention = BertAttention(config, dtype, param_dtype, precision, rngs=rngs)
        self.intermediate = BertIntermediate(
            config, dtype, param_dtype, precision, rngs=rngs
        )
        self.output = BertOutput(config, dtype, param_dtype, precision, rngs=rngs)

    def __call__(
        self,
        hidden_states: jax.Array,
        attention_mask: jax.Array | None = None,
    ) -> jax.Array:
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
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
        self.layer = [
            BertLayer(config, dtype, param_dtype, precision, rngs=rngs)
            for _ in range(config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states: jax.Array,
        attention_mask: jax.Array | None = None,
    ) -> jax.Array:
        for layer in self.layer:
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states


class BertPooler(nnx.Module):
    def __init__(
        self,
        config: BertConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: tp.Optional[lax.Precision] = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.dense = nnx.Linear(
            self.config.hidden_size,
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, hidden_states):
        cls_hidden_state = hidden_states[:, 0]
        cls_hidden_state = self.dense(cls_hidden_state)
        return nnx.tanh(cls_hidden_state)


class BertPretrainedModel(NNXPretrainedModel, HuggingFaceCompatible):
    hf_config_class = BertConfig 
    hf_model_class = transformers.BertModel
    embedding_layer_names = [
        "word_embeddings",
        "position_embeddings",
        "token_type_embeddings",
    ]
    layernorm_names = ["layer_norm", "LayerNorm"]


class BertModel(BertPretrainedModel):

    def __init__(
        self,
        config: BertConfig,
        dtype: jnp.dtype = jnp.float32,  # the dtype of the computation
        param_dtype: jnp.dtype = jnp.float32,
        precision: tp.Optional[lax.Precision] = None,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(config, dtype, param_dtype, precision, rngs=rngs)
        self.embeddings = BertEmbeddings(
            config, dtype, param_dtype, precision, rngs=rngs
        )
        self.encoder = BertEncoder(config, dtype, param_dtype, precision, rngs=rngs)
        self.pooler = BertPooler(config, dtype, param_dtype, precision, rngs=rngs)

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        token_type_ids: jax.Array | None = None,
        position_ids: jax.Array | None = None,
    ) -> jax.Array:

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), jnp.shape(input_ids)
            )

        embeddings = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        hidden_states = self.encoder(
            embeddings,
            attention_mask,
        )
        return hidden_states

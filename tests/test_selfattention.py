import jax
import numpy as np
import pytest
from transformers.models.bert.modeling_bert import BertSelfAttention as TFBertSelfAttention
from src.models.bert.configuration_bert import BertConfig
from src.models.bert.modeling_bert import  BertSelfAttention
from flax import nnx
from tests.utils import assert_close, move_model_weight
import torch


# def test_self_attention():
#     config = BertConfig()
#     tf_model = TFBertSelfAttention(config).eval()
#     devices = jax.devices(backend="cpu")[0]
#
#     model = BertSelfAttention(config, rngs=nnx.Rngs(0))
#     move_model_weight(model, tf_model)
#
#     attention_head_size = config.hidden_size // config.num_attention_heads
#     all_head_size = attention_head_size * config.num_attention_heads 
#     tf_input = torch.ones((1, config.hidden_size, all_head_size))
#     input = jax.numpy.ones((1, config.hidden_size, all_head_size))
#     assert_close(model.query(input), tf_model.query(tf_input))



# def test_assert():
#     assert_close(np.array([[1, 2, 3], [1, 1, 1]]), np.array([[1, 2, 4], [10, 100, 1000]]))

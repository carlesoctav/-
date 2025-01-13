import flax
from torch import unflatten

from src.models.bert.configuration_bert import BertConfig
from src.models.bert.modeling_bert import BertModel
from flax import nnx
from flax.nnx import traversals
import transformers



model = BertModel(BertConfig(axis_dims = (1,4,1,1) , num_hidden_layers=1), rngs = nnx.Rngs(0, dropout = 100, params = 10))

state, others = nnx.state(model, nnx.Param, ...)
print(f"DEBUGPRINT[3]: test_model.py:16: others={others}")



# model_transformers = transformers.BertModel(transformers.BertConfig(num_hidden_layers=1))
# embedding_layer_names = [
#         "word_embeddings",
#         "position_embeddings",
#         "token_type_embeddings",
#     ]
# layernorm_names = ["layer_norm", "LayerNorm"]
#
# state_dict = model_transformers.state_dict()
# print(f"DEBUGPRINT[4]: test_model.py:22: state_dict={state_dict}")
# flax_dict = convert_dict_state_to_flax_params(state_dict, embedding_layer_names=embedding_layer_names, layernorm_names=layernorm_names)
# print(f"DEBUGPRINT[3]: test_model.py:23: flax_dict={flax_dict}")
# unflatten_dict = traversals.unflatten_mapping(flax_dict) 
#
#
# graph, param, others = nnx.split(model, nnx.Param, ...)
# param.replace_by_pure_dict(unflatten_dict)
# print(f"DEBUGPRINT[2]: test_model.py:25: param={param}")

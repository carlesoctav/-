from flax.nnx import traversals 
from flax import nnx
from transformers import AutoTokenizer
import transformers
from src.models.bert.modeling_bert import BertModel
from src.models.bert.configuration_bert import BertConfig
import jax.numpy as jnp

from src.utils.paramaters_transformations import convert_torch_state_to_flax_state, recreate_rngs_state_for_lazy_model

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
a = tokenizer("hallo", return_tensors = "jax")
b = tokenizer("hallo", return_tensors = "pt")
config = BertConfig(num_hidden_layers=1)
config._attn_implementation = "eager"

hf_model = transformers.BertModel(config)
print(f"DEBUGPRINT[33]: test_hf_2.py:18: hf_model.dtype={hf_model.dtype}")
hf_model.eval()
state_dict = hf_model.state_dict()


model = BertModel.lazy_init(
config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=nnx.Rngs(0)
)

flatten_params = convert_torch_state_to_flax_state(state_dict, model)
unflatten_params = traversals.unflatten_mapping(flatten_params)

params, others = nnx.state(model, nnx.Param, ...)
params.replace_by_pure_dict(unflatten_params)
others = recreate_rngs_state_for_lazy_model(others)
nnx.update(model, params, others)
model.eval()

model_from_jax = model(**a)
print(f"DEBUGPRINT[4]: test_hf_2.py:36: model_from_jax={model_from_jax}")

model_from_torch = hf_model(**b)
print(f"DEBUGPRINT[5]: test_hf_2.py:39: model_from_torch={model_from_torch}")





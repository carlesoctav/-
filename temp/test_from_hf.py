from flax.nnx import traversals 
from flax import nnx
from transformers import AutoTokenizer
import transformers
from src.models.bert.modeling_bert import BertModel
from src.models.bert.configuration_bert import BertConfig

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
a = tokenizer("hallo", return_tensors = "jax")
b = tokenizer("hallo", return_tensors = "pt")
print(f"DEBUGPRINT[4]: test_from_hf.py:9: a={a}")
print(f"DEBUGPRINT[5]: test_from_hf.py:11: b={b}")
model = BertModel.from_huggingface("google-bert/bert-base-uncased", rngs = nnx.Rngs(0))
model.eval()
hf_model = transformers.BertModel.from_pretrained("google-bert/bert-base-uncased")
hf_model.eval()


model_from_jax = model(**a)
print(f"DEBUGPRINT[1]: test_from_hf.py:15: model_from_jax={model_from_jax}")

model_from_torch = hf_model(**b)

print(f"DEBUGPRINT[2]: test_from_hf.py:17: model_from_torch={model_from_torch}")




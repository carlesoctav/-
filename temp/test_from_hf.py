from flax.nnx import traversals 
from flax import nnx
from transformers import AutoTokenizer
from src.models.bert.modeling_bert import BertModel
from src.models.bert.configuration_bert import BertConfig

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
a = tokenizer("hallo", return_tensors = "jax")
print(f"DEBUGPRINT[4]: test_a.py:7: a={a}")
model = BertModel.from_huggingface("google-bert/bert-base-uncased", rngs = nnx.Rngs(0))


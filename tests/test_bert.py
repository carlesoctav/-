import jax.numpy as jnp
from flax import nnx
import transformers
from transformers import AutoTokenizer

from src.models.bert.modeling_bert import BertModel
from src.models.bert.configuration_bert import BertConfig
from tests.utils import assert_close, move_model_weight



def test_bert():
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    a = tokenizer("hallo", return_tensors="jax")
    b = tokenizer("hallo", return_tensors="pt")
    config = BertConfig(num_hidden_layers=1)
    config._attn_implementation = "eager"
    hf_model = transformers.BertModel(config)
    hf_model.eval()

    model = BertModel.lazy_init(
        config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=nnx.Rngs(0)
    )
    move_model_weight(model, hf_model)
    model.eval()

    output1 = model(**a)
    output2 = hf_model(**b)
    assert_close(output1, output2.last_hidden_state)

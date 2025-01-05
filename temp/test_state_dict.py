import transformers
from transformers import BertModel, BertConfig
from transformers.modeling_flax_pytorch_utils import convert_pytorch_state_dict_to_flax
from src.utils.convert_params import convert_dict_state_to_flax_params


model = BertModel(BertConfig(num_hidden_layers=1))
a = model.state_dict()
print(len(a.keys()))

convert_dict_state_to_flax_params(a)



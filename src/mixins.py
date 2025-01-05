import typing as tp
import jax.numpy as jnp
from jax import lax
from transformers import PretrainedConfig, PreTrainedModel
import gc
from abc import abstractmethod
from flax import nnx
from flax.nnx import traversals
from src.configuration_utils import NNXPretrainedConfig
from src.utils.convert_params import convert_dict_state_to_flax_params


if tp.TYPE_CHECKING:
    from src.modeling_utils import NNXPretrainedModel


class HuggingFaceCompatible:
    hf_config_class: tp.Type[NNXPretrainedConfig]
    hf_model_class: tp.Type[PreTrainedModel]
    embedding_layer_names: tp.List[str]
    layernorm_names: tp.List[str]

    @classmethod
    @abstractmethod
    def lazy_init(cls, *args, **kwargs) -> "NNXPretrainedModel":
        raise NotImplementedError

    @classmethod
    def from_huggingface(
        cls,
        pretrained_model_name_or_path: str,
        dtype: jnp.dtype = jnp.float32,  
        param_dtype: jnp.dtype = jnp.float32,
        precision: tp.Optional[lax.Precision] = None,
        *model_args,
        rngs: nnx.Rngs,
        do_partition: bool = False,
        **kwargs,
    ):
        try:
            import torch

            if torch.cuda.is_available():

                def _clear():
                    gc.collect()
                    torch.cuda.empty_cache()

            else:

                def _clear():
                    gc.collect()

        except ModuleNotFoundError as er:
            raise ModuleNotFoundError(
                "in order to load model from torch you should install torch first "
                "run `pip install torch`"
            ) from er

        hf_config = cls.hf_config_class.from_pretrained(pretrained_model_name_or_path)
        hf_model = cls.hf_model_class.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

        state_dict = hf_model.state_dict()
        del hf_model
        flatten_params = convert_dict_state_to_flax_params(
            state_dict, cls.embedding_layer_names, cls.layernorm_names
        )
        model = cls.lazy_init(config=hf_config, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        model, params, others = nnx.split(model, nnx.Param, ...)
        unflatten_params = traversals.unflatten_mapping(flatten_params) 
        print(f"DEBUGPRINT[8]: mixins.py:72: unflatten_params={unflatten_params}")
        params.replace_by_pure_dict(unflatten_params)

        return model

import typing as tp
import jax.numpy as jnp
from jax import lax
from transformers import PreTrainedModel
import gc
from abc import abstractmethod
from flax import nnx
from flax.nnx import traversals
from src.configuration_utils import NNXPretrainedConfig
from src.utils.paramaters_transformations import convert_torch_state_to_flax_state


if tp.TYPE_CHECKING:
    from src.modeling_utils import NNXPretrainedModel


class HuggingFaceCompatible:
    hf_config_class: tp.Type[NNXPretrainedConfig]
    hf_model_class: tp.Type[PreTrainedModel]

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
        do_shard: bool = False,
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

        model = cls.lazy_init(
            config=hf_config, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

        flatten_params = convert_torch_state_to_flax_state(state_dict, model)
        unflatten_params = traversals.unflatten_mapping(flatten_params)

        params = nnx.state(model, nnx.Param)
        params.replace_by_pure_dict(unflatten_params)
        nnx.update(model, params)

        if do_shard:
            model.shard_model()

        return model

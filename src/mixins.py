import typing as tp
from transformers import PretrainedConfig, PreTrainedModel
import gc
from abc import abstractmethod
from flax import nnx


if tp.TYPE_CHECKING:
    from src.modeling_utils import NNXPretrainedModel
    from src.configuration_utils import NNXPreTrainedConfig 

class HuggingFaceCompatible:
    hf_config_class: tp.Type[PretrainedConfig]
    hf_model_class: tp.Type[PreTrainedModel]

    @classmethod
    @abstractmethod
    def lazy_init(cls, *args, **kwargs) -> "NNXPretrainedModel": 
        raise NotImplementedError

    @classmethod
    def from_huggingface(
        cls,
        pretrained_model_name_or_path: str,
        do_partition: bool = False,
        *model_args,
        config: tp.Optional["NNXPreTrainedConfig"]  = None,
        **kwargs
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
        hf_model = cls.hf_model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)

        state_dict = hf_model.state_dict()
        del hf_model
        # model = cls.lazy_init(config=hf_config, do_partition=do_partition, **kwargs)



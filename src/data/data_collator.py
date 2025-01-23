from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase
import typing as tp



def numpy_collate():
    pass


@dataclass
class DataCollatorForLanguageModeling:
    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: tp.Optional[int] = None
    tf_experimental_compile: bool = False


    def __call__(self, batch):
        pass



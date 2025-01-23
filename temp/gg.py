import grain.python as pygrain
from datasets import load_dataset
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")


def tokenize(batch, max_length=128):
    a = tokenizer(
        batch["id_abs"],
        max_length=max_length,
        return_tensors="jax",
        padding="max_length",
        truncation=True,
    )
    input_ids = a["input_ids"]
    print(f"DEBUGPRINT[4]: gg.py:10: input_ids={input_ids.shape}")
    print(f"DEBUGPRINT[3]: gg.py:10: input_ids={input_ids}")
    print(f"DEBUGPRINT[2]: gg.py:10: input_ids={type(input_ids)}")
    return a


ds = load_dataset("carlesoctav/skripsi_UI_membership_30K", split="train[:100]")

# ds = ds.map(tokenize, batched=True)
# ds = ds.with_format(type = "numpy")
print(f"DEBUGPRINT[6]: gg.py:22: ds={ds}")


class Tokenize(pygrain.MapTransform):

    def __init__(self, max_length=128, return_tensors="np", padding="max_length", truncation=True):
        self.max_length = max_length
        self.return_tensors = return_tensors
        self.padding = padding
        self.truncation = truncation

    def map(self, element):
        print(f"DEBUGPRINT[3]: gg.py:47: element={element}")
        a = tokenizer(
            element["id_abs"],
            max_length=self.max_length,
            return_tensors=self.return_tensors,
            padding=self.padding,
            truncation=self.truncation,
        )
        return a 


sampler = pygrain.IndexSampler(
    len(ds), shuffle=True, shard_options=pygrain.NoSharding(), seed=42
)
pygrain.BatchOperation
transformations = [Tokenize(), pygrain.Batch(batch_size=2)]

loader = pygrain.DataLoader(data_source=ds, sampler=sampler, operations=transformations, worker_count =0)

for batch in loader:
    print(batch.shape)
    break

print(f"DEBUGPRINT[5]: gg.py:24: loader={loader}")

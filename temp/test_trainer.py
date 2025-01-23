import jax.numpy as jnp
from tqdm import tqdm
import numpy as np
import typing as tp
import chex
from src.trainer import TrainerUtil
import jax
from flax import nnx
import optax
from torch.utils import data



def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class XORDataset(data.Dataset):

    def __init__(self, size, seed, std=0.1):
        """
        Inputs:
            size - Number of data points we want to generate
            seed - The seed to use to create the PRNG state with which we want to generate the data points
            std - Standard deviation of the noise (see generate_continuous_xor function)
        """
        super().__init__()
        self.size = size
        self.np_rng = np.random.RandomState(seed=seed)
        self.std = std
        self.generate_continuous_xor()

    def generate_continuous_xor(self):
        # Each data point in the XOR dataset has two variables, x and y, that can be either 0 or 1
        # The label is their XOR combination, i.e. 1 if only x or only y is 1 while the other is 0.
        # If x=y, the label is 0.
        data = self.np_rng.randint(low=0, high=2, size=(self.size, 2)).astype(np.float32)
        label = (data.sum(axis=1) == 1).astype(np.int32)
        # To make it slightly more challenging, we add a bit of gaussian noise to the data points.
        data += self.np_rng.normal(loc=0.0, scale=self.std, size=data.shape)

        self.data = data
        self.label = label

    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data_point = self.data[idx]
        data_label = self.label[idx]
        return data_point, data_label


class SimplerClassifier(nnx.Module):

    def __init__(self, num_inputs, num_hidden, num_outputs, rngs=nnx.Rngs(0)):
        self.linear1 = nnx.Linear(num_inputs, num_hidden, rngs=rngs)
        self.linear2 = nnx.Linear(num_hidden, num_outputs, rngs=rngs)

    def __call__(self, x):
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.linear2(x)
        return x


class ClassificationTrainer(TrainerUtil):
    def loss_function(self, model: nnx.Module, batch):
        inputs, labels = batch
        logits = model(inputs).squeeze(axis=-1) 
        print(f"DEBUGPRINT[10]: test_trainer.py:78: labels={labels.shape}")
        print(f"DEBUGPRINT[8]: test_trainer.py:79: logits={logits.shape}")
        loss = optax.sigmoid_binary_cross_entropy(logits, labels)
        print(f"DEBUGPRINT[9]: test_trainer.py:81: loss={loss.shape}")
        pred_labels = (logits > 0).astype(jnp.float32)
        acc = (pred_labels == labels).mean()
        loss = loss.mean()
        return loss, acc


model = SimplerClassifier(2, 10, 1)
classification_trainer = ClassificationTrainer()
model, optimizer = classification_trainer.setup(model, optax.sgd(learning_rate=0.1), debug= True)

for name, leaf in nnx.iter_graph(model):
    if isinstance(leaf, nnx.Param):
        print(name)
        jax.debug.visualize_array_sharding(leaf)

dataset = XORDataset(500, seed = 123)
dataloader = data.DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=numpy_collate)
dataloader = classification_trainer.setup_dataloader(dataloader)

step = 0
for epoch in range(100):
    for batch in tqdm(dataloader):
        if step == 0:
            jax.debug.visualize_array_sharding(batch)
        (loss, acc), grads = classification_trainer.train_step(model, optimizer, batch)
        print(f"Epoch {epoch}: Loss = {loss}")
        print(f"DEBUGPRINT[7]: test_trainer.py:106: acc={acc}")

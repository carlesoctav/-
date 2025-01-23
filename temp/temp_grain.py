import grain.python as pygrain
import numpy as np
from torchvision.datasets import MNIST
import jax


data_dir = "./data"
jax.make_array_from_process_local_data


class Dataset:
    def __init__(self, data_dir, train=True):
        self.data_dir = data_dir
        self.train = train
        self.load_data()

    def load_data(self):
        self.dataset = MNIST(self.data_dir, download=True, train=self.train)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return np.ravel(np.array(img, dtype=np.float32)), label


mnist_dataset = Dataset(data_dir)
batch_size = 10

sampler = pygrain.SequentialSampler(
    num_records=len(mnist_dataset),
    shard_options=pygrain.ShardOptions()

data_loader =pygrain.DataLoader(
        data_source=mnist_dataset,
        sampler=sampler,
        operations=[pygrain.Batch(batch_size)],
)


for batch in data_loader:
    print(batch[0])
    break

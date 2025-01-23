import jax
import typing as tp 
from torch.utils.data import DataLoader
from jax.sharding import PartitionSpec, Mesh, NamedSharding
import numpy as np

class _JaxDataLoader:
    def __init__(self, dataloader: DataLoader, sharding: NamedSharding | None):
        self.__dict__.update(dataloader.__dict__)
        self._dataloader = dataloader
        self._named_sharding = sharding 
        self._num_iter_calls = 0


    def __len__(self) -> int:
        return len(self._dataloader)

    def __iter__(self) -> tp.Union[tp.Iterator[tp.Any], tp.Generator[tp.Any, None, None]]:
        self._num_iter_calls += 1

        if self._named_sharding is None: 
            yield from iter(self._dataloader)
        else:
            for item in self._dataloader:
                yield jax.device_put(item, self._named_sharding)



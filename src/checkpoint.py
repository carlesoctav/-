from flax import nnx
import typing as tp


class CheckpointManager:

    @staticmethod
    def load_from_orbax(model: nnx.Module, path:str, do_sharding: bool):
        pass

    @staticmethod
    def load_from_torch(model: nnx.Module,path:tp.Union[str, tp.List[str]], do_sharding: bool):
        pass

    def load_from_safe_tensor(model: nnx.Module, path:str, do_sharding: bool):
        pass


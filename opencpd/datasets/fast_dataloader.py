import threading
import torch
import queue
from torch.utils.data import DataLoader


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super().__init__(**kwargs)
        self.stream = torch.cuda.Stream(
            local_rank
        )  # create a new cuda stream in each process
        self.local_rank = local_rank
        # self.custom_collect_fn = custom_collect_fn
        
    def __iter__(self):
        self.iter = super().__iter__()
        self.preload()
        return self


    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None

        with torch.cuda.stream(self.stream): # 将数据预先放进gpu
            for key, val in self.batch.items():
                if type(val) == torch.Tensor:
                    self.batch[key] = val.to(
                            device=self.local_rank, non_blocking=True
                        )


    def __next__(self):
        torch.cuda.current_stream().wait_stream(
            self.stream
        )  # wait tensor to put on GPU
        batch = self.batch
        # batch = self.custom_collect_fn(self.batch)
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


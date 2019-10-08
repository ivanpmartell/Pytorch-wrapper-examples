import numpy as np
from dataclasses import dataclass
from torch.utils.data import Dataset


@dataclass(eq=False)
class FuzzBuzzDataset(Dataset):
    input_size: int
    start: int = 0
    end: int = 1000

    def encoder(self, num):
        ret = [int(i) for i in '{0:b}'.format(num)]
        return [0] * (self.input_size - len(ret)) + ret

    def __getitem__(self, idx):
        x = self.encoder(idx)
        if idx % 15 == 0:
            y = 0
        elif idx % 5 == 0:
            y = 1
        elif idx % 3 == 0:
            y = 2
        else:
            y = 3
        return np.array(x, dtype=np.float32), np.array(y, dtype=np.int64)

    def __len__(self):
        return self.end - self.start

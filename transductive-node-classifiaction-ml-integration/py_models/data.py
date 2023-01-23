import torch
from torch import Tensor
from typing import Dict


class SplitIndex:
    def __init__(self, train: Tensor, valid: Tensor, test: Tensor):
        self.train = train
        self.valid = valid
        self.test = test


class Data:
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'

    def __init__(self, x: Tensor, y: Tensor, train_idx: Tensor, valid_idx: Tensor, test_idx: Tensor):
        self.x: Tensor = x
        self.y: Tensor = y
        self.split_idx: Dict[str, Tensor] = {Data.TRAIN: train_idx, Data.VALID: valid_idx, Data.TEST: test_idx}

    def train_idx(self):
        return self.split_idx[Data.TRAIN]

    def valid_idx(self):
        return self.split_idx[Data.VALID]

    def test_idx(self):
        return self.split_idx[Data.TEST]

    def x_train(self):
        return self.x[self.split_idx[Data.TRAIN]]

    def x_valid(self):
        return self.x[self.split_idx[Data.VALID]]

    def x_test(self):
        return self.x[self.split_idx[Data.TEST]]

    def y_train(self):
        return self.y[self.split_idx[Data.TRAIN]]

    def y_valid(self):
        return self.y[self.split_idx[Data.VALID]]

    def y_test(self):
        return self.y[self.split_idx[Data.TEST]]

    def _x_slice(self, split: str):
        return self.x[self.split_idx[split]]

    def _y_slice(self, split: str):
        return self.y[self.split_idx[split]]

    def _apply(self, d, func, *args, **kwargs):
        res = dict()
        for k, v in self.split_idx.items():
            res[k] = func(d[v], *args, **kwargs)
        return res

    def x_apply(self, func, *args, **kwargs):
        return self._apply(self.x, func, *args, **kwargs)

    def y_apply(self, func, *args, **kwargs):
        return self._apply(self.y, func, *args, **kwargs)

    def split_sizes(self):
        res = dict()
        for k, v in self.split_idx.items():
            res[k] = v.shape[0]
        return res

    def to(self, device: torch.device, non_blocking: bool = False):
        x = self.x.to(device=device, non_blocking=non_blocking)
        y = self.y.to(device=device, non_blocking=non_blocking)
        split_idx = dict()
        for k, v in self.split_idx.items():
            split_idx[k] = v.to(device=device, non_blocking=non_blocking)
        return Data(x, y, split_idx[Data.TRAIN], split_idx[Data.VALID], split_idx[Data.TEST])

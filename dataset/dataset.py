import torch
import os
from pathlib import Path
from torch.utils.data import DataLoader

DATA_DIR = Path(os.path.dirname(os.path.realpath(__file__)), 'raw')


class InstanceStandardize:
    def __init__(self):
        pass

    def fit(self, dataset):
        self.mean = dataset.tensors[1].mean([1,2]).unsqueeze(-1).unsqueeze(-1)
        self.std = dataset.tensors[1].std([1,2]).unsqueeze(-1).unsqueeze(-1)

    def transform(self, dataset):
        y = dataset.tensors[1]
        y = y - self.mean
        y = y / self.std
        dataset.tensors = (dataset.tensors[0], y)
        return dataset

    def fit_transform(self, data):
        self.fit(data)
        self.transform(data)


class DatasetStandardize:
    def __init__(self, axis=[0,1,2]):
        self.axis = axis
        pass

    def fit(self, dataset):
        self.mean = dataset.tensors[1][self.axis].mean()
        self.std = dataset.tensors[1][self.axis].std()

    def transform(self, dataset):
        y = dataset.tensors[1]
        y = y - y.mean()
        y = y / self.std
        dataset.tensors = (dataset.tensors[0], y)
        return dataset

    def fit_transform(self, data):
        self.fit(data)
        self.transform(data)


class Dataset:
    def __init__(self):
        pass

    def train_loader(self, batch_size):
        return DataLoader(self.train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

    def eval_loader(self, batch_size):
        return DataLoader(self.eval_dataset,
                          batch_size=batch_size,
                          shuffle=True)

    @property
    def y_dim(self,):
        return self._y_dim

    def load(self, path):
        dataset_path = Path(path, "dataset")
        self.__dict__.update(torch.load(dataset_path).__dict__)

    def save(self, path):
        dataset_path = Path(path, "dataset")
        torch.save(self, dataset_path)

    def normalize(self,):
        self.scaler = DatasetStandardize()
        self.scaler.fit_transform(self.train_dataset)
        self.scaler.transform(self.eval_dataset)

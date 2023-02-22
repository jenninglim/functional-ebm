from .dataset import Dataset, DATA_DIR, InstanceStandardize
from pathlib import Path
import pandas as pd
import numpy as np
import torch


class Gridwatch(Dataset):
    def __init__(self,
                 seed=1,
                 train_portion=0.7,
                 n_steps=2,
                 pre_load=False):
        FILE_DIR = Path(DATA_DIR, 'Gridwatch/processed_gridwatch.csv')

        df = pd.read_csv(FILE_DIR)
        df = df.set_index([' timestamp'])
        df.columns = pd.to_datetime(df.columns)
        df = df.iloc[::n_steps] # reduces the sampling points by n_steps times
        y_full = torch.tensor(df.values.T, dtype=torch.float).unsqueeze(-1)
        x_full = torch.arange(df.shape[0],  dtype=torch.float).unsqueeze(-1) #/ df.shape[0]
        n_train = int(y_full.shape[0] * train_portion)
        train_indices = np.random.choice(df.shape[1], size=n_train, replace=False)
        eval_indices = np.setdiff1d(np.arange(y_full.shape[0]), train_indices)
        if pre_load:
            self.train_dataset = torch.load(DATA_DIR / ".."/ "gridwatch" / "train_dataset")
            self.eval_dataset = torch.load(DATA_DIR / ".."/ "gridwatch" / "eval_dataset")
        else:
            self.train_dataset = torch.utils.data.TensorDataset(x_full.repeat(n_train, 1, 1),
                                                                y_full[train_indices])
            self.eval_dataset = torch.utils.data.TensorDataset(x_full.repeat(y_full.shape[0] - n_train, 1, 1),
                                                                y_full[eval_indices])
        self.x = x_full
        self.train_dataset.x = self.x
        self.eval_dataset.x = self.x
        self._y_dim = 1
        self.normalize()

    def __str__(self):
        return "Gridwatch"

    def normalize(self):
        self.scaler = InstanceStandardize()
        self.scaler.fit_transform(self.train_dataset)
        self.scaler.fit_transform(self.eval_dataset)

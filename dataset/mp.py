import torch
from sktime.datasets import load_from_arff_to_dataframe
from pathlib import Path
from .dataset import Dataset, DATA_DIR
import numpy as np


class Melbourne(Dataset):
    def __init__(self, seed=1, train_portion=0.7, pre_load=False):
        FILE_DIR = Path(DATA_DIR, 'melborunepedestrian/MelbournePedestrian_TRAIN.arff')
        data = load_from_arff_to_dataframe(FILE_DIR)
        samples = []
        for i in range(data[0].shape[0]):
            samples.append(torch.tensor(data[0]['dim_0'][i]))
        y_full = torch.stack(samples, axis=0)
        ind = torch.isnan(y_full).sum(axis=1) == 0
        y_full = y_full[ind].float()
        x_full = torch.arange(y_full.shape[1],  dtype=torch.float).unsqueeze(-1)
        n_train = int(y_full.shape[0] * train_portion)
        train_indices = np.random.choice(y_full.shape[0], size=n_train, replace=False)
        eval_indices = np.setdiff1d(np.arange(y_full.shape[0]), train_indices)
        if pre_load:
            self.train_dataset = torch.load(DATA_DIR / ".."/ "melbourne" / "train_dataset")
            self.eval_dataset = torch.load(DATA_DIR / ".."/ "melbourne" / "eval_dataset")
        else:
            self.train_dataset = torch.utils.data.TensorDataset(x_full.repeat(n_train, 1, 1),
                                                                y_full[train_indices].unsqueeze(-1))
            self.eval_dataset = torch.utils.data.TensorDataset(x_full.repeat(y_full.shape[0] - n_train, 1, 1),
                                                                y_full[eval_indices].unsqueeze(-1))
            self.normalize()

        self.x = x_full
        self.train_dataset.x = self.x
        self.eval_dataset.x = self.x
        self._y_dim = 1

from .dataset import Dataset, DATA_DIR
from pathlib import Path
import os
import pandas as pd
import torch


class FTSE(Dataset):
    def __init__(self,
                 n_samples=400,
                 train_portion=0.5,
                 n_points=200,
                 desired_column='high',
                 normalise=True,
                 seed=1,
                 pre_load=False):
        index_column = 'date'
        FILE_DIR = Path(DATA_DIR, 'individual_stocks_5yr')
        files = []
        for i, file_name in enumerate(os.listdir(FILE_DIR)):
            df = pd.read_csv(Path(FILE_DIR, file_name), index_col=index_column)[desired_column]
            files.append(df)
            df.name = str(i)
        data = pd.DataFrame(files).T
        data.index = pd.to_datetime(data.index)
        data = data.dropna(axis=1)
        n_points = min(n_points, data.shape[1])

        y_full = torch.tensor(data.values.T, dtype=torch.float)[:, :n_points].unsqueeze(-1)
        x_full = torch.arange(n_points,  dtype=torch.float).unsqueeze(-1) / 5
        n_train = int(n_samples * train_portion)
        n_eval = n_samples - n_train
        if pre_load:
            self.train_dataset = torch.load(DATA_DIR / ".."/ "stocks" / "train_dataset")
            self.eval_dataset = torch.load(DATA_DIR / ".."/ "stocks" / "eval_dataset")
        else:
            self.train_dataset = torch.utils.data.TensorDataset(x_full.repeat(n_train, 1, 1),
                                                                y_full[:n_train])
            self.eval_dataset = torch.utils.data.TensorDataset(x_full.repeat(n_samples - n_train, 1, 1),
                                                                y_full[n_train:n_train+n_eval])
            self.normalize()

        self.x = x_full
        self.train_dataset.x = self.x
        self.eval_dataset.x = self.x
        self._y_dim = 1

    def __str__(self):
        return "FTSE"

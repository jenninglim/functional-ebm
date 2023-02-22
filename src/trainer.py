import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
import pickle
from shutil import copyfile
from math import floor
from tqdm.auto import tqdm
import wandb
from copy import deepcopy


class Trainer:
    def __init__(self,
                 model,
                 optimiser,
                 dataset,
                 cd,
                 scheduler=None,
                 checkpoint=False,
                 p=0.2,
                 save_dir=None,
                 batch_size=128,
                 use_wandb=False):
        self.model = model
        self.optimiser = optimiser
        eval_size = int(p * len(dataset))
        self.dataset = dataset

        # Split dataset into train-eval according to split variable p.
        self.train_dataset = TensorDataset(*dataset[eval_size:])
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True)
        self.eval_dataset = TensorDataset(*dataset[:eval_size])
        self.eval_loader = DataLoader(self.eval_dataset,
                                      batch_size=batch_size,
                                      shuffle=True)
        self.scheduler = scheduler
        self.cd = cd
        self.best_model = None
        self.checkpoint = checkpoint
        self.save_dir = save_dir
        self.train_losses = {'loss': [],
                             'joint': [],
                             'cond': []}
        self.eval_losses = {'loss': [],
                            'joint': [],
                            'cond': []}
        self.use_wandb = use_wandb

    def train(self,
              patience: int = 10,
              n_epochs: int = 100):
        """
        :param patience: early stopping parameter.
        :param n_epochs: number of epochs.
        """
        min_loss_epoch = 0
        min_loss = np.float('inf')
        for epoch in (pbar := tqdm(range(n_epochs))):
            self.model.train()
            train_stats, eval_stats = self.step()
            pbar.set_description(f"Train loss {train_stats['loss']:.3f}"
                                 f"Eval loss {eval_stats['loss']:.3f}")
            self.model.eval()

            if self.checkpoint and self.save_dir is not None:
                torch.save(self.model, Path(self.save_dir, "model_" + str(epoch)))

            if min_loss > eval_stats['loss']:
                min_loss_epoch = epoch
                min_loss = eval_stats['loss']
                self.best_model = deepcopy(self.model)
                if self.save_dir is not None:
                    # source path
                    model_path = Path(self.save_dir, "model_" + str(epoch))

                    # destination path
                    dest_model_path = self.save_dir / ".." / "final_model"

                    # copy
                    copyfile(model_path, dest_model_path)

            # Update wandb with training statistics.
            if self.use_wandb:
                train_eval_stats = {**{f"train_{k}": v for k, v in train_stats.items()},
                                    **{f"eval_{k}": v for k, v in eval_stats.items()}}
                wandb.log({**train_eval_stats})
            plt.close('all')

            if patience != 0:
                if min_loss_epoch + patience < epoch:
                    logging.info("Early stopping.")
                    break

        # Save train losses and eval losses.
        if self.save_dir is not None:
            path = Path(self.save_dir, "train_losses")
            path.touch()
            with path.open('wb') as f:
                pickle.dump(self.train_losses, f)
            if self.eval_loader is not None:
                path = Path(self.save_dir, "eval_losses")
                path.touch()
                with path.open('wb') as f:
                    pickle.dump(self.eval_losses, f)
        if self.use_wandb:
            wandb.save(self.save_dir.resolve(), policy="now")

    def load(self,):
        # load_losses
        train_loss_path = Path(self.save_dir, "train_losses")
        if train_loss_path.is_file():
            with train_loss_path.open("rb") as f:
                self.train_losses = pickle.load(f)
        if self.eval_loader is not None:
            eval_loss_path = Path(self.save_dir, "eval_losses")
            if eval_loss_path.is_file():
                with eval_loss_path.open("rb") as f:
                    self.eval_losses = pickle.load(f)

    def step(self):
        # train
        train_loss = {'loss': 0.,
                      'cond': 0.,
                      'joint': 0.}
        self.model.train()
        for i, data in enumerate(self.train_loader):
            self.optimiser.zero_grad()

            x, y = data
            x = x.to(device=self.model.device)
            if self.model.encoder is not None:
                x = self.model.encoder(x)
            y = y.to(device=self.model.device)
            p = floor(np.random.rand(1) * (x.shape[1] - 1)) + 1
            o = np.random.choice(x.shape[1], size=p)
            loss, cond, joint = self.cd(x, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.param_model.parameters(),
                                           100.,
                                           norm_type='inf')
            torch.nn.utils.clip_grad_norm_(self.model.energy.parameters(),
                                           100.,
                                           norm_type='inf')

            train_loss['loss'] = train_loss['loss'] + loss.item() / len(self.train_loader)
            train_loss['cond'] = train_loss['cond'] + cond / len(self.train_loader)
            train_loss['joint'] = train_loss['joint'] + joint / len(self.train_loader)
            self.optimiser.step()

            if torch.isnan(loss):
                raise Exception("Loss is nan.")
        # eval
        if self.eval_loader is not None:
            self.model.eval()
            eval_loss = {'loss': 0.,
                         'cond': 0.,
                         'joint': 0.}
            for data in self.eval_loader:
                x, y = data
                x = x.to(device=self.model.device)
                if self.model.encoder is not None:
                    x = self.model.encoder(x)
                y = y.to(device=self.model.device)
                loss, cond, joint = self.cd(x, y)
                eval_loss['loss'] = eval_loss['loss'] + loss.item() / len(self.eval_loader)
                eval_loss['joint'] = eval_loss['joint'] + joint / len(self.eval_loader)
                eval_loss['cond'] = eval_loss['cond'] + cond / len(self.eval_loader)
                if torch.isnan(loss):
                    raise Exception("Loss is nan.")

            logging.info(f"Eval Loss: energy_data {eval_loss['cond']: .3f}, energy_model {eval_loss['joint']: .3f}, loss {eval_loss['loss']: .3f}")
            self.eval_losses['loss'].append(eval_loss['loss'])
            self.eval_losses['cond'].append(eval_loss['cond'])
            self.eval_losses['joint'].append(eval_loss['joint'])

        logging.info(f"Train Loss: energy_data {train_loss['cond']: .3f}, energy_model {train_loss['joint']: .3f}, total {train_loss['loss']: .3f}")
        self.train_losses['loss'].append(train_loss['loss'])
        self.train_losses['joint'].append(train_loss['joint'])
        self.train_losses['cond'].append(train_loss['cond'])

        self.scheduler.step(eval_loss['loss'])
        return train_loss, eval_loss

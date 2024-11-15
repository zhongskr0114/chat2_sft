# trainer
# --base.py
from abc import ABC, abstractmethod

import tqdm
from torch.optim import Optimizer
from torch import nn
class Trainer(ABC):
    def __init__(self, optimizer:Optimizer, model:nn.Module, max_epochs:int=2):
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.epoch_bar = tqdm.trange(self.max_epochs, desc='epochs', disable = False)

    @abstractmethod
    def _train(self, epoch):
        raise NotImplementedError

    @abstractmethod
    def _eval(self, epoch):
        raise NotImplementedError
    @abstractmethod
    def _before_fit(self):
        raise NotImplementedError
    def fit(self, *args, **kwargs):
        self._before_fit(*args, **kwargs)
        for epoch in range(1, self.max_epochs+1):
            self._train(epoch)
            self._eval(epoch)
            self.epoch_bar.update()
            print(f'第{epoch}伦训练完成。')


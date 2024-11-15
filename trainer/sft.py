# trainer
# --sft.py

from typing import Optional,Any

import tqdm
from torch.optim import Optimizer
from torch.utils._pytree import tree_map
import torch
from torch import nn
from trainer.base import Trainer


def to_device(x:Any, device):
    def _to(t: Any):
        if isinstance(t, torch.Tensor):
            return t.to(device)
        elif isinstance(t, torch.nn.Module):
            return t.to(device)
        return t
    return tree_map(_to, x)

class SFTTrainer(Trainer):
    def __init__(self, optimizer:Optimizer, model:nn.Module, lr_scheduler,device,
                 batch_size:int=2, max_epochs:int =2):
        super(SFTTrainer, self).__init__(optimizer=optimizer, model=model, max_epochs=max_epochs)
        self.scheduler = lr_scheduler
        self.batch_size = batch_size
        self.device = device


    def _before_fit(self, train_dataloader, eval_dataloader):
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.total_loss = 0

    def _train(self, epoch):
        self.model = to_device(self.model, self.device)
        self.model.train()
        for batch_id, batch in enumerate(self.train_dataloader):
            batch = to_device(batch, self.device)
            outputs = self.model(
                batch['input_ids'],
                attention_mask = batch['attention_mask'],
                labels = batch['labels']
                )
            loss = outputs.loss
            loss.backward()
            self.total_loss += loss.item()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            if batch_id%100==0:
                print(f'第{epoch}轮次：第{batch} step,loss={loss}')
        print(f'一共训练了{batch_id}steps')

    def _eval(self, epoch:int):
        if self.eval_dataloader is not None:
            self.model.eval()
            with torch.no_grad():
                loss_sum,num_seen=0,0
                for batch in self.eval_dataloader:
                    batch = to_device(batch, self.device)
                    outputs = self.model(batch['input_ids'],
                                         attention_mask = batch['attention_mask'],
                                         labels = batch['labels'])
                    loss = outputs.loss
                    loss_sum += loss.item()
                    num_seen += batch['input_ids'].size(0)
                loss_mean = loss_sum/num_seen
                print(f'loss_mean={loss_mean}')
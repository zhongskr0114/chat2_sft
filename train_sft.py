import argparse
from typing import Type

from torch import optim
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
import torch
from torch.utils.data import DataLoader
import math
from dataset.sft_dataset import SupervisedDataset, CollateForSurpervisedDataset
from trainer.sft import SFTTrainer


def train(args):

    # pretrained model and tokenizer load...
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.pretrain,
                                              trust_remote_code=True,
                                              cache_dir = args.cache)
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=args.pretrain,
                                                 trust_remote_code=True,
                                                 cache_dir = args.cache)
    # tokenizer + pad token ?
    tokenizer.pad_token = tokenizer.eos_token

    # optimizer,train_data,eval_data,lr_scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_dataset = SupervisedDataset(data_path=args.dataset, tokenizer=tokenizer, max_len=args.max_len)
    eval_dataset = None
    data_collator = CollateForSurpervisedDataset(tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=data_collator, pin_memory = True)
    eval_dataloader = None

    step_per_epoch = len(train_dataloader)
    max_steps = math.ceil(args.max_epochs * step_per_epoch)

    lr_scheduler = get_scheduler('cosine', optimizer,
                                 num_warmup_steps=math.ceil(max_steps * 0.03),
                                 num_training_steps=max_steps
                                 )

    # device
    device = 'cpu'

    # instantiate and call trainer
    trainer = SFTTrainer(model = model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                         max_epochs=2, device=device, batch_size=args.batch_size)
    trainer.fit(train_dataloader = train_dataloader,eval_dataloader=eval_dataloader)

    # save weights
    state = model.state_dict()
    torch.save(state, args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', type=str, default='facebook/opt-125m')
    parser.add_argument('--dataset', type=str, default='ds/ds_mini.jsonl')
    parser.add_argument('--cache', type=str, default='D:/cache')
    parser.add_argument('--save_path', type=str, default='output')
    parser.add_argument('--max_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--lr', type=float, default=5e-6)
    args = parser.parse_args()
    print(type(args))
    print(args)
    train(args)
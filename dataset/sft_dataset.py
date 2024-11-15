# dataset
# -- sft_dataset.py
"""
PROMPT_DICT:
Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
[指令内容]
### Input(.opt):

### Response:

"""
from attr.validators import max_len
from torch.utils.data import Dataset
from utils import jsonl_load
from transformers import PreTrainedTokenizer

PROMPT_DICT  = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
from typing import List,Dict

from typing import List, Dict
from transformers import PreTrainedTokenizer
import copy


def preprocess(sources: List[str], targets: List[str], tokenizer: PreTrainedTokenizer, max_len: int) -> Dict:
    # 将 sources 和 targets 合并，每个样本的目标序列将计算损失
    examples = [s + t for s, t in zip(sources, targets)]

    # 对合并后的序列进行分词
    tokenized_examples = tokenizer(examples, padding='max_length', truncation=True, max_length=max_len,
                                   return_tensors="pt")
    tokenized_sources = tokenizer(sources, padding='max_length', truncation=True, max_length=max_len,
                                  return_tensors="pt")

    # 获取 input_ids 和创建 labels 副本
    input_ids = tokenized_examples['input_ids']
    labels = copy.deepcopy(input_ids)

    # 将 source 部分的 labels 设置为 -100，以忽略它们的损失
    for i, src_len in enumerate(tokenized_sources['attention_mask'].sum(dim=1)):
        labels[i, :src_len] = -100  # 忽略 source 部分的 token

    # 返回包含 input_ids 和 labels 的字典
    return {
        'input_ids': input_ids,
        'labels': labels
    }


class SupervisedDataset(Dataset):
    def __init__(self, data_path:str, tokenizer:PreTrainedTokenizer, max_len:int = 512):
        super(SupervisedDataset,self).__init__()
        # load 数据
        list_data_dict = jsonl_load(data_path)

        # format数据
        prompt_input = PROMPT_DICT['prompt_input']
        prompt_no_input = PROMPT_DICT['prompt_no_input']
        sources = [
            prompt_input.format_map(example)
           if example.get('input') is not None
           else prompt_no_input.format_map(example)
           for example in list_data_dict
        ]
        targets = [
            f"{example['output']}{tokenizer.eos_token}"
            for example in list_data_dict
        ]
        data_dict = preprocess(sources, targets, tokenizer, max_len)
        self.input_ids = data_dict['input_ids']
        self.labels = data_dict['labels']

    def __getitem__(self, item):
        return dict(input_ids = self.input_ids[item], labels = self.labels[item])
    def __len__(self):
        return len(self.input_ids)

from dataclasses import dataclass
import torch
@dataclass
class CollateForSurpervisedDataset:
    tokenizer:PreTrainedTokenizer
    def __call__(self, instances: List[Dict])->Dict[str, torch.tensor]:
        input_ids = [instance['input_ids'] for instance in instances]
        labels = [instance['labels'] for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True,
                                                 padding_value=-100)
        return dict(input_ids = input_ids, labels = labels, attention_mask = input_ids.ne(self.tokenizer.pad_token_id))

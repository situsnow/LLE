
import numpy as np
import torch
import random
import collections

from utils.constants import MEM_GRP_CRTR_CONFIG, MEM_GRP_CRTR_LBL

"""
Copied from https://github.com/h3lio5/episodic-lifelong-learning.git
Code for the paper: Episodic Memory in Lifelong Language Learning (https://arxiv.org/pdf/1906.01076v3.pdf) 
for the text classification setup. 
"""


class ReplayMemory(object):
    """
        Create the empty memory buffer for pretraining
    """

    def __init__(self, buffer=None):

        total_keys = 0
        if buffer is None:
            self.memory = collections.OrderedDict()
        else:
            self.memory = buffer
            total_keys = len(buffer.keys())
            # convert the keys from np.bytes to np.float32
            self.all_keys = np.frombuffer(
                np.asarray(list(self.memory.keys())), dtype=np.float32).reshape(total_keys, 768)
        self.max_len = total_keys

    def len(self):
        return len(self.memory)

    def group_indices_by_value(self, criteria):
        # idx=2 for label, idx=3 for config_idx
        groups = {}
        loc = 2 if criteria == MEM_GRP_CRTR_LBL else 3

        all_keys = list(self.memory)
        for k in all_keys:
            value = self.memory[k][loc]
            try:
                groups[value].append(k)
            except KeyError:
                groups[value] = [k]
        return groups

    def push(self, keys, examples, max_memory, config_idx, criteria=MEM_GRP_CRTR_CONFIG):
        """
        Add the examples as key-value pairs to the memory dictionary with content,attention_mask,label tuple as value
        and key determined by key network
        """
        contents, attn_masks, labels, raw_text = examples
        # update the memory dictionary
        for i, key in enumerate(keys):
            # the size should not exceed defined max_memory per category and induce randomness
            if random.uniform(0, 1) < 0.5:
                query = labels[i] if criteria == MEM_GRP_CRTR_LBL else config_idx
                group_of_memory = self.group_indices_by_value(criteria)
                try:
                    if len(self.memory) > 0 and len(group_of_memory[query]) >= max_memory:
                        # remove a random example belongs to current config or label from memory
                        del_key = random.choice(group_of_memory[query])
                        del self.memory[del_key]
                except KeyError:
                    print(f'There ain\'t any memory from config/label {query} yet.')
                # numpy array cannot be used as key since it is non-hashable,
                # hence convert it to bytes to use as key
                self.memory.update(
                    {key.tobytes(): (contents[i], attn_masks[i], labels[i], config_idx, raw_text[i])})
                if contents[i].size > self.max_len:
                    self.max_len = contents[i].size

    def sample(self, sample_size):
        if len(self.memory) < sample_size:
            sample_size = len(self.memory)
        if type(self.memory) == collections.OrderedDict:
            keys = random.sample(list(self.memory), sample_size)
        else:
            keys = random.sample(range(len(self.memory)), sample_size)
        # pad in case the lengths of instances in the sample are not the same
        contents = np.array([np.pad(self.memory[k][0], (0, self.max_len - self.memory[k][0].size), 'constant')
                             for k in keys])
        attn_masks = np.array([np.pad(self.memory[k][1], (0, self.max_len - self.memory[k][1].size), 'constant')
                               for k in keys])
        labels = np.array([self.memory[k][2] for k in keys])
        config_indices = np.array([self.memory[k][3] for k in keys])

        return torch.LongTensor(contents), torch.LongTensor(attn_masks), torch.LongTensor(labels), config_indices

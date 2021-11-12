import random
import numpy as np
import torch

from sklearn.cluster import KMeans

from fairseq.fairseq import utils


def sample_from_memory(args, task, max_tokens):

    # select select those shorter than the max_tokens
    candidates = np.where(task.dataset(args.memory_split).src_sizes <= max_tokens)[0].tolist()

    selected_mem = random.sample(candidates, min(len(candidates), args.memory_sample_size))

    mini_batch = [task.dataset(args.memory_split)[i] for i in selected_mem]

    mem_samples = task.dataset(args.memory_split).collater(mini_batch, max_tokens)
    return mem_samples


def concat_train_mem(train_samples, mem_samples):
    train_samples['id'] = torch.cat((train_samples['id'], mem_samples['id']), 0)
    train_samples['nsentences'] = train_samples['nsentences'] + mem_samples['nsentences']
    train_samples['ntokens'] = train_samples['ntokens'] + mem_samples['ntokens']
    # dict
    new_net_input = dict()
    new_net_input['src_tokens'] = torch.cat((train_samples['net_input']['src_tokens'],
                                             mem_samples['net_input']['src_tokens']), 0)
    new_net_input['src_lengths'] = torch.cat((train_samples['net_input']['src_lengths'],
                                              mem_samples['net_input']['src_lengths']), 0)
    new_net_input['src_text'] = None
    train_samples['net_input'] = new_net_input

    train_samples['target'] = torch.cat((train_samples['target'], mem_samples['target']), 0)

    if torch.cuda.is_available():
        train_samples = utils.move_to_cuda(train_samples)

    return [train_samples]
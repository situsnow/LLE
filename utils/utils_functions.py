import os
import sys
import re
import math
import torch
from pathlib import Path


def read_plain_file(path):
    data = []
    with open(path) as f:
        for each in f:
            data.append(each.strip())
    return data


def write_plain_file(path, data):
    with open(path, 'w') as f:
        for each in data:
            f.write(each)


def convert_to_list(var):
    if type(var) == list:
        return var
    elif type(var) == float:
        return [var]
    else:
        raise RuntimeError(f"The parameter {var} is neither a list or a float.")


def handle_task_specific_fairseq_parameters(parm, value, opt='append'):
    try:
        i = sys.argv.index(parm)
        if opt == 'append':
            sys.argv[i + 1] = os.path.join(sys.argv[i + 1], value)
        elif opt == 'replace':
            # replace
            sys.argv[i+1] = value
        else:
            # assert
            try:
                assert sys.argv[i+1] == value
            except AssertionError:
                print(f'{parm} is not equal to the defined value {value}. Replace and keep running.')
                sys.argv[i+1] = value
    except ValueError:
        raise RuntimeError(f'{parm} is not in current fairseq explainer parameter.')


def copy_dict(args, task):
    dict_file = os.path.join(args.explainer_log_dir, args.dict_file)
    dir_path = os.path.join(args.explainer_log_dir, task)
    new_dict_path = os.path.join(dir_path, args.new_dict_file)

    os_cmd = f"cp {dict_file} {new_dict_path}"
    os.system(os_cmd)
    print(f'Copy the dictionary to current task directory with command: {os_cmd}')


def remove_dict(args, task):
    dir_path = os.path.join(args.explainer_log_dir, task)
    dict_file = os.path.join(dir_path, args.new_dict_file)

    os_cmd = f'rm {dict_file}'
    os.system(os_cmd)

    print(f'Remove the dictionary in current directory with command {os_cmd}')


def clean_str(string, lower=True):
    """
    Tokenization/string cleaning
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """

    # special handling for web documents
    string = re.sub(r"<br />", " ", string)

    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[^A-Za-z(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    # remove non English words by nltk wordnet
    # from nltk.corpus import wordnet
    if lower:
        return string.strip().lower()
    else:
        return string.strip()


def send_to_cuda(a_tensor):
    if torch.cuda.is_available():
        return a_tensor.cuda()
    else:
        return a_tensor


def join_sentence(x):
    if type(x) == list:
        return " ".join(x)
    else:
        return x


def split_sentence(x):
    if type(x) == list:
        return x
    else:
        return x.split(" ")


def save_to_disk(x_list, exp_list, src_file, tgt_file, rewrite=False):

    parent_dir = os.path.dirname(src_file)
    Path(parent_dir).mkdir(parents=True, exist_ok=True)

    opt = 'w' if rewrite or not os.path.exists(src_file) else 'a'
    with open(src_file, opt) as f:
        for x in x_list:
            f.write(x + '\n')
            # f.write('\n')
        f.flush()
    opt = 'w' if rewrite or not os.path.exists(tgt_file) else 'a'
    with open(tgt_file, opt) as f:
        for y in exp_list:
            f.write(",".join(list(map(str, y))) + '\n')
            # f.write('\n')
        f.flush()


def normalize(items, norm, a=-1, b=1):
    denom = 1.0  # no normalization by default
    if norm is not None:
        denom = sum(items)  # otherwise default denominator is sum
    if norm == 'max':
        denom = max(items)
    elif norm == 'sum_square':
        denom = sum([x ** 2 for x in items])
    elif norm == 'range':
        min_x = min(items)
        max_x = max(items)
        items = [(b - a) * ((x - min_x) / (max_x - min_x)) + a for x in items]
        return items
    elif norm == 'tanh':
        # convert items into tensor
        items = torch.tanh(torch.tensor(items))
        items = items.detach().cpu().tolist()
        return items

    items = [x / denom if denom != 0.0 else x for x in items]
    return items


def log_odds(prob):
    return math.log(prob / (1 - prob)) if prob < 1.0 else 0.0










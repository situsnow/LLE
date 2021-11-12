import os
import torch
import torch.utils.data as data
from datasets import load_dataset, ReadInstruction
from utils.constants import INF, TEST, AMSRV, REVERSE, TRAIN, DEV
from utils.utils_functions import read_plain_file, split_sentence

TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4


class MyAMRDataset(data.Dataset):

    def __init__(self, encodings):
        self.raw_text = encodings['review_body']
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']
        self.labels = encodings['star_rating']

    def __getitem__(self, idx):
        item = dict()
        # item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['raw_text'] = self.raw_text[idx]
        item['attention_mask'] = torch.tensor(self.attention_mask[idx])
        item['input_ids'] = torch.tensor(self.input_ids[idx])
        item['labels'] = torch.tensor(0 if self.labels[idx]<3 else 1)  # 0: neg; 1: pos
        return item

    def __len__(self):
        return len(self.labels)

    def get_input_ids(self):
        return self.input_ids

    def get_attention_masks(self):
        return self.attention_mask

    def get_labels(self):
        return self.labels

    def get_raw_texts(self):
        return self.raw_text


def get_data_files(args, task):
    data_files = {INF: (os.path.join(args.explainer_log_dir, task, args.fs_inf_src),
                        os.path.join(args.explainer_log_dir, task, args.fs_inf_tgt)),
                  TEST: (os.path.join(args.explainer_log_dir, task, args.fs_test_src),
                         os.path.join(args.explainer_log_dir, task, args.fs_test_tgt))}
    return data_files


def handle_task_order(args, task_order):
    dataset = args.dataset

    dataset_config = huggingface_dataset_config(dataset)

    if task_order == REVERSE:
        dataset_config.reverse()
    return dataset_config


def huggingface_dataset_config(dataset):
    config_dict = dict()
    config_dict[AMSRV] = ['Home_v1_00', 'Outdoors_v1_00', 'Wireless_v1_00', 'Music_v1_00', 'Books_v1_00',
                          'Office_Products_v1_00', 'Luggage_v1_00', 'Sports_v1_00', 'Jewelry_v1_00',
                          'Video_Games_v1_00']

    return config_dict[dataset]


def get_dataset_categories(dataset):
    dataset_categories = dict()
    dataset_categories[AMSRV] = ['neg', 'pos']
    return dataset_categories[dataset]


def get_split_dataloader(args, eval_config, tokenizer):
    train_loader, valid_loader, test_loader = load_amr_data(args, eval_config, tokenizer)
    if args.eval_cat == TRAIN:
        return train_loader
    elif args.eval_cat == DEV:
        return valid_loader
    else:
        return test_loader


def get_task_display_name(tasks):
    new_tasks = []
    # max_char = 10
    for task_name in tasks:
        if task_name[-1] == '0':
            task_name = task_name.replace('_v1_00', '')
        else:
            task_name = task_name.replace('_v1_0', '')
        # new_tasks.append('\n'.join(task_name[i:i+max_char] for i in range(0, len(task_name), max_char)))
        new_tasks.append(task_name)
    return new_tasks


def load_amr_data(args, config, tokenizer=None):

    train_data = load_amr_data_split(args, config, 0, int(args.split_ratio[0]), TRAIN_BATCH_SIZE, tokenizer,
                                     shuffle=True)
    valid_data = load_amr_data_split(args, config, int(args.split_ratio[0]),
                                     int(args.split_ratio[0]) + int(args.split_ratio[1]),
                                     VALID_BATCH_SIZE, tokenizer)
    test_data = load_amr_data_split(args, config, int(args.split_ratio[0]) + int(args.split_ratio[1]),
                                    int(args.split_ratio[0]) + int(args.split_ratio[1]) + int(args.split_ratio[2]),
                                    TEST_BATCH_SIZE, tokenizer)
    return train_data, valid_data, test_data


def load_amr_data_split(args, config, from_ratio, to_ratio, batch_size, tokenizer, shuffle=False):
    def tokenize(batch):
        return tokenizer(batch['review_body'], padding=True, truncation=True)
    ri = ReadInstruction('train', from_=from_ratio, to=to_ratio, unit=args.split_unit)
    dataset = load_dataset(args.dataset, config, split=ri, cache_dir=args.cache_dir,
                                 ignore_verifications=True)

    # filter review with rating 3
    dataset = dataset.filter(lambda x, indice: x['star_rating'] != 3, with_indices=True)

    if tokenizer is not None:
        dataset = dataset.map(tokenize, batched=True, batch_size=len(dataset))
        dataset = MyAMRDataset(dataset)
        params = {'batch_size': batch_size, 'shuffle': shuffle}
        # params = {'batch_size': batch_size}
        dataloader = data.DataLoader(dataset, **params)
        return dataloader
    else:
        return dataset


def load_src_tgt(src_path, tgt_path, clean=True, filter_len=False):
    src = read_plain_file(src_path)
    tgt = read_plain_file(tgt_path)

    try:
        assert len(src) == len(tgt)
    except AssertionError:
        raise RuntimeError('The lengths of src and tgt do not match.')
    if clean:
        new_src, new_tgt = [], []
        i = 0
        while i < len(src):
            x = src[i].strip().split(" ")
            if filter_len and (len(x) < 3 or len(x) > 250):   # filter too long/short documents
                continue
            y = [float(w) for w in tgt[i].strip().split(",")]

            new_src.append(x)
            new_tgt.append(y)

            i += 1

        return new_src, new_tgt
    else:
        return src, tgt


def get_dataset_load_func(dataset):
    load_func = dict()
    load_func[AMSRV] = load_amr_data
    return load_func[dataset]


def load_plain_memory(file_paths, column_names, tokenizer=None):
    def tokenize(batch):
        return tokenizer(batch[column_names[1]], padding=True, truncation=True)
    # for amazon product review, the column_names should be ['star_rating', 'review_body']
    if len(file_paths) == 0:
        file_paths = file_paths[0]
    dataset = load_dataset('csv', data_files=file_paths,
                           column_names=column_names)
    plain_dataset = dataset[TRAIN]    # get the dataset from the default split

    if tokenizer is not None:
        plain_dataset = plain_dataset.map(tokenize, batched=True, batch_size=len(plain_dataset))

        plain_dataset = MyAMRDataset(plain_dataset)

        mem_params = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True}
        mem_dataloader = data.DataLoader(plain_dataset, **mem_params)
        return mem_dataloader
    else:
        return plain_dataset


def clean_data(text, lower=True):
    import re
    from utils.utils_functions import clean_str
    non_word = re.compile(r'(\W+)|$').match

    text = [x for x in list(filter(None, re.split(r'(\W+)|$', clean_str(text, lower)))) if not non_word(x)]
    return text


def get_train_data(train_dataloader, max_len=300):

    clean_text = map(clean_data, train_dataloader.dataset.get_raw_texts())
    clean_short_text = list(filter(lambda x: len(x) > max_len, clean_text))

    return clean_short_text


def count_eligible_examples(X, max_len):
    count = 0
    for x in X:
        if 3 < len(split_sentence(x)) < max_len:
            count += 1
    return count
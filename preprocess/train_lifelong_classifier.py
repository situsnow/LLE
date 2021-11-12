import argparse, sys, os, random
from pathlib import Path
import numpy as np
from tqdm import tqdm
import copy
import pickle
import matplotlib.pyplot as plt

import torch
from datasets.utils.info_utils import UnexpectedSplits, ExpectedMoreSplits, NonMatchingSplitsSizesError
from transformers import AdamW

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from utils.constants import *

from preprocess.ReplayMemory import ReplayMemory
from utils.utils_functions import load_tokenizer_keyencoder, load_model, get_dataset_load_func, \
    define_text_classification_label_orders, handle_task_order


# Defining some key variables that will be used later on in the training
LEARNING_RATE = 1e-05
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
SPLIT_VALIDATION = 'Validation'
SPLIT_TEST = 'Test'


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def set_model_parameters():
    parser = argparse.ArgumentParser(description='Fine-tune BERT as black-box model')
    # path
    parser.add_argument('--data-dir', type=str, default='/home/xsit1/snow/dataset/')
    parser.add_argument('--cache-dir', type=str, default='/home/xsit1/snow/cache/')
    parser.add_argument('--log-dir', type=str, default='bert_base_models', help='the logging dir for the trained model')
    parser.add_argument('--print-stdout', action='store_true', help='default print stdout to file')

    # [HuggingFace] Amazon product review dataset
    parser.add_argument('--dataset', type=str, default=AMSRV, choices=[TEXT_CLASSIFICATION, AMSRV])
    parser.add_argument('--split-ratio', nargs='+', default=[20000, 2000, 2000], help='the ratio for the dataset')
    parser.add_argument('--split-unit', type=str, default=HUGGINGFACE_SPLIT_ABS,
                        choices=[HUGGINGFACE_SPLIT_ABS, HUGGINGFACE_SPLIT_PERCENT])
    parser.add_argument('--start-idx', type=int, default=0, help='the start index of dataset config')
    parser.add_argument('--end-idx', type=int, default=-1, help='the last index of dataset config')
    parser.add_argument('--task-order', type=str, choices=[DIRECT, RANDOM, REVERSE, TC_I, TC_II, TC_III, TC_IV],
                        default=DIRECT,
                        help='the ordering of the task, need to save random')
    # parser.add_argument('--reverse', action='store_true', help='to reverse the order of tasks, default False.')

    # model
    parser.add_argument('--arch', type=str, default=DISTILBERT, help='the architecture of the model')
    parser.add_argument('--wd', type=float, default=0.0, help='the weight decay (l2) in AdamW optimizer')

    # algo - replay
    parser.add_argument('--no-replay', action='store_true', help='indication to replay memory, '
                                                                 'default False (will replay)')
    parser.add_argument('--replay-freq', type=int, default=500, help='the replay frequency of steps')
    parser.add_argument('--max-memory', type=int, default=64,
                        help='the max number of examples of each config/labels in memory.')
    parser.add_argument('--sample-size', type=int, default=64, help='the sample size from all memory in replay.')
    parser.add_argument('--sample-strategy', type=str, default=MEMORY_RANDOM,
                        choices=[MEMORY_RANDOM, MEMORY_CLUSTER, MEMORY_GSS])
    parser.add_argument('--cluster-selection', type=str,
                        help='The strategy to select example from cluster')
    # algo - train
    parser.add_argument('--save-interval', type=int, default=1000, help='the interval to evaluate and save checkpoints')
    parser.add_argument('--max-epoch', type=int, default=1, help='the maximum epochs to train the classifier')

    args, _ = parser.parse_known_args(sys.argv)

    args.pretrained_path = "%s/%s/pretrained_%s/" % (args.data_dir, args.dataset, args.arch)
    args.pretrained_encoder_path = "%s/%s/pretrained_%s_encoder/" % (args.data_dir, args.dataset, args.arch)
    args.log_dir = '%s/%s/%s/' % (args.data_dir, args.dataset, args.log_dir)

    args.dataset_config = handle_task_order(args, args.task_order)
    if args.end_idx == -1:
        args.end_idx = len(args.dataset_config)

    return args


def calculate_accuracy(preds, targets):
    n_correct = (preds == targets).sum().item()
    correct_indices = (preds == targets).detach().clone()
    return n_correct, correct_indices


def get_keys(key_encoder, ids, attn_masks):
    """
    Copied from https://github.com/h3lio5/episodic-lifelong-learning.git
    Code for the paper: Episodic Memory in Lifelong Language Learning (https://arxiv.org/pdf/1906.01076v3.pdf)
    for the text classification setup.
    :return: key representation of the documents
    """
    with torch.no_grad():
        last_hidden_states = key_encoder(
            ids, attention_mask=attn_masks)
    # Obtain key representation of every text content by selecting the its [CLS] hidden representation
    keys = last_hidden_states[0][:, 0, :]

    return keys


def train(args, epoch, training_loader, dev_loader, test_loader,
          device, model, tokenizer, optimizer, memory, config_idx, key_encoder):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0

    # train_loss = []
    for step, batch in tqdm(enumerate(training_loader, 0)):
        model.train()  # in case evaluation happen on previous step
        batch_cp = copy.deepcopy(batch)
        del batch

        if memory.len() != 0 and (step+1) % args.replay_freq == 0 and not args.no_replay:
            # get examples from memory
            ids, attn_masks, targets, _ = \
                memory.sample(sample_size=args.sample_size)

            def replay(start_index, end_index):
                # print("======== Starting replay. ========")
                tmp_ids = ids[start_index:end_index]
                tmp_attn_masks = attn_masks[start_index:end_index]
                tmp_targets = targets[start_index:end_index]
                if device == CUDA:
                    tmp_ids, tmp_attn_masks, tmp_targets = tmp_ids.cuda(), tmp_attn_masks.cuda(), tmp_targets.cuda()

                results = model(tmp_ids, tmp_attn_masks, labels=tmp_targets, return_dict=True)
                loss, logits = results.loss, results.logits
                optimizer.zero_grad()
                loss.backward()
                # train_loss.append(loss)
                # # When using GPU
                optimizer.step()
                # Update tracking variables
                max_val, max_idx = torch.max(logits.data, dim=1)
                replay_loss = loss.item()
                replay_acc, _ = calculate_accuracy(max_idx, tmp_targets)
                del tmp_ids
                del tmp_targets
                del tmp_attn_masks
                del loss
                return replay_loss, replay_acc
            start = 0
            while start * TRAIN_BATCH_SIZE < ids.size(0):
                end = min((start + 1) * TRAIN_BATCH_SIZE, ids.size(0))
                replay_tr_loss, replay_n_correct = replay(TRAIN_BATCH_SIZE * start, end)
                tr_loss += replay_tr_loss
                n_correct += replay_n_correct
                start += 1

            nb_tr_examples += ids.size(0)
            nb_tr_steps += 1

            del ids
            del attn_masks
            del targets

        ids = batch_cp['input_ids'].to(device, dtype=torch.long)
        attn_masks = batch_cp['attention_mask'].to(device, dtype=torch.long)
        targets = batch_cp['labels'].to(device, dtype=torch.long)
        raw_text = batch_cp['raw_text']   # the raw text is only for saving it into disk for LLE memory replay

        results = model(ids, attn_masks, labels=targets, return_dict=True)
        loss, logits = results.loss, results.logits
        tr_loss += loss.item()
        y_hat = torch.argmax(logits.data, dim=1)
        #
        cor_cnt, _ = calculate_accuracy(y_hat, targets)
        n_correct += cor_cnt

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del loss

        # save current examples
        if not args.no_replay:
            keys = get_keys(key_encoder, ids, attn_masks)
            memory.push(keys.cpu().numpy(),
                        (ids.cpu().numpy(), attn_masks.cpu().numpy(), targets.cpu().numpy(), raw_text),
                        args.max_memory, config_idx,
                        criteria=MEM_GRP_CRTR_LBL if args.dataset == TEXT_CLASSIFICATION else MEM_GRP_CRTR_CONFIG,
                        strategy=args.sample_strategy, cluster_selection=args.cluster_selection)
            del keys
        del ids
        del attn_masks
        del targets

    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

    # evaluate on all previous config
    print(f'==== Start validating on previous data configs at the end of epoch {epoch+1} ====')
    for i in range(0, config_idx):
        config = args.dataset_config[i]
        print(f'Start validating on {config}')
        _, prev_dev_dataloader, prev_test_dataloader = get_dataset_load_func(args.dataset)(args, config, tokenizer)
        if prev_dev_dataloader is not None:
            valid(args, model, prev_dev_dataloader, None, device, current_config=i, split=SPLIT_VALIDATION)
        valid(args, model, prev_test_dataloader, None, device, current_config=i, split=SPLIT_TEST)

    print(f'Start validating on {args.dataset_config[config_idx]}')
    if dev_loader is not None:
        valid(args, model, dev_loader, None, device, current_config=config_idx, split=SPLIT_VALIDATION)
    valid(args, model, test_loader, None, device, current_config=config_idx, split=SPLIT_TEST)
    save_checkpoint(args, model, config_idx)
    sys.stdout.flush()


def valid(args, model, testing_loader, memory, device, current_config, split):
    model.eval()
    n_correct, va_loss, nb_va_steps, nb_va_examples = 0, 0, 0, 0
    n_correct_per_config = {}
    n_per_config = {}

    def evaluate(ids, mask, targets):
        results = model(ids, mask, labels=targets, return_dict=True)
        loss, logits = results.loss, results.logits
        y_hat = torch.argmax(logits.data, dim=1)
        acc_cnt, acc_indices = calculate_accuracy(y_hat, targets)
        return loss.item(), acc_cnt, acc_indices

    def insert_dict(dict_obj, key, value):
        try:
            prev_val = dict_obj[key]
            dict_obj[key] = prev_val + value
        except KeyError:
            dict_obj[key] = value

    with torch.no_grad():
        for step, batch in tqdm(enumerate(testing_loader, 0)):
            batch_cp = copy.deepcopy(batch)
            del batch
            ids = batch_cp['input_ids'].to(device, dtype=torch.long)
            mask = batch_cp['attention_mask'].to(device, dtype=torch.long)
            targets = batch_cp['labels'].to(device, dtype=torch.long)

            loss, acc, _ = evaluate(ids, mask, targets)
            va_loss += loss
            n_correct += acc
            nb_va_steps += 1
            nb_va_examples += targets.size(0)

            del batch_cp
            del ids
            del mask
            del targets

        n_per_config[current_config] = nb_va_examples
        n_correct_per_config[current_config] = n_correct

        # validate on all memory
        if memory is not None:
            ids, attn_masks, targets, config_indices = memory.sample(sample_size=memory.len())
            unique, counts = np.unique(config_indices, return_counts=True)
            # save the num of instances in each config in memory
            for k,v in dict(zip(unique, counts)).items():
                insert_dict(n_per_config, k, v)
            start_index = 0
            while start_index * TRAIN_BATCH_SIZE < ids.size(0):
                end_index = min((start_index + 1) * TRAIN_BATCH_SIZE, ids.size(0))
                tmp_ids = ids[start_index * TRAIN_BATCH_SIZE:end_index]
                tmp_attn_masks = attn_masks[start_index * TRAIN_BATCH_SIZE:end_index]
                tmp_targets = targets[start_index * TRAIN_BATCH_SIZE:end_index]
                if device == CUDA:
                    tmp_ids, tmp_attn_masks, tmp_targets = tmp_ids.cuda(), tmp_attn_masks.cuda(), tmp_targets.cuda()
                loss, acc, acc_indices = evaluate(tmp_ids, tmp_attn_masks, tmp_targets)
                va_loss += loss
                n_correct += acc
                nb_va_steps += 1
                nb_va_examples += tmp_targets.size(0)
                del tmp_ids
                del tmp_attn_masks
                del tmp_targets

                # save acc according to config in the dataset
                cp_acc_indices = acc_indices.detach().cpu().clone().numpy()
                memory_acc = config_indices[start_index * TRAIN_BATCH_SIZE:end_index][cp_acc_indices]
                unique, counts = np.unique(memory_acc, return_counts=True)
                for k,v in dict(zip(unique, counts)).items():
                    insert_dict(n_correct_per_config, k, v)
                del acc_indices
                start_index += 1

    epoch_loss = va_loss / nb_va_steps
    epoch_accu = (n_correct * 100) / nb_va_examples

    print(f"{split} Avg Loss: {epoch_loss}")
    print(f"{split} Avg Accuracy: {epoch_accu}")
    print(f"{split} Avg Accuracy on each config: ", end="")
    keylist = n_correct_per_config.keys()
    for k in sorted(keylist):
        config = args.dataset_config[k]
        v = n_correct_per_config[k]
        print(f"{config} : {v * 100/n_per_config[k]}", end=", ")
    print("")
    sys.stdout.flush()

    return epoch_accu


def save_train_loss(args, config_idx, train_loss):
    """
    Function to save the image of training loss v/s iterations graph
    """
    plt.figure(figsize=(15, 8))
    plt.title("Training loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.plot(train_loss)

    image_dir = '%s/%s' % (args.log_dir, args.dataset_config[config_idx])
    Path(image_dir).mkdir(parents=True, exist_ok=True)

    plt.savefig(image_dir+'/train_loss.png')


def save_checkpoint(args, model, config_idx, step=None, memory=None):
    suffix = '_last' if step is None else str(step)

    folder = CHECKPOINT_PREFIX + suffix

    checkpoint_path = '%s/%s/%s' % (args.log_dir, args.dataset_config[config_idx], folder)
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    print(f'Saving the checkpoint on step {suffix}.')
    model.save_pretrained(checkpoint_path)
    if memory is not None:
        memory_pkl = 'memory_' + suffix + '.pkl'
        with open(checkpoint_path + '/' + memory_pkl, 'wb') as f:
            pickle.dump(memory, f)


def reinitiate_memory_by_config(args, tokenizer, key_encoder, device):
    memory = ReplayMemory()
    # reinitiate memory if args.start_idx is not 0
    for i in range(0, args.start_idx):
        config = args.dataset_config[i]

        try:
            train_dataloader, _, _ = get_dataset_load_func(args.dataset)(args, config, tokenizer)

            size = len(train_dataloader.dataset.labels)
            selected_idx = random.sample(range(size), args.max_memory)

            selected_ids = np.array(train_dataloader.dataset.get_input_ids())[selected_idx]
            selected_attn_masks = np.array(train_dataloader.dataset.get_attention_masks())[selected_idx]
            selected_labels = np.array(train_dataloader.dataset.get_labels())[selected_idx]
            selected_raw_text = np.array(train_dataloader.dataset.get_raw_texts())[selected_idx]

            # convert the labels, aligned with lifelonglearning.utils_functions.MyDataset.__getitem__
            selected_labels = np.array([0 if i < 3 else 1 for i in selected_labels])

            start = 0
            while start * TRAIN_BATCH_SIZE < args.max_memory:
                end = min((start + 1) * TRAIN_BATCH_SIZE, args.max_memory)
                tmp_ids = torch.tensor(selected_ids[start * TRAIN_BATCH_SIZE:end, :], device=device)
                tmp_attn_masks = torch.tensor(selected_attn_masks[start * TRAIN_BATCH_SIZE:end, :], device=device)
                tmp_raw_texts = selected_raw_text[start * TRAIN_BATCH_SIZE: end]

                tmp_keys = get_keys(key_encoder, tmp_ids, tmp_attn_masks)
                memory.push(tmp_keys.cpu().numpy(),
                            (tmp_ids.cpu().numpy(), tmp_attn_masks.cpu().numpy(),
                            selected_labels[start * TRAIN_BATCH_SIZE:end], tmp_raw_texts),
                            args.max_memory, i,
                            criteria=MEM_GRP_CRTR_CONFIG,
                            strategy=args.sample_strategy)
                start += 1
                del tmp_ids
                del tmp_attn_masks
                del tmp_raw_texts
                del tmp_keys

            del selected_ids
            del selected_attn_masks
            del selected_labels
            del selected_raw_text
            del _
            del train_dataloader
        except (NonMatchingSplitsSizesError or ExpectedMoreSplits or UnexpectedSplits) as e:
            print(e)
            print(f'============ Skip the configuration from {args.dataset_config[i]}. ============')
            # do not remove the configuration here in case we need to verify in main process again
            continue

    return memory


def reinitiate_memory_by_label(args, tokenizer, key_encoder, device):
    memory = ReplayMemory()
    # reinitiate memory if args.start_idx is not 0
    for i in range(0, args.start_idx):
        config = args.dataset_config[i]

        try:
            train_dataloader, _, _ = get_dataset_load_func(args.dataset)(args, config, tokenizer)

            unique_labels = set(train_dataloader.dataset.labels)
            all_labels = np.array(train_dataloader.dataset.labels)
            for y in unique_labels:
                all_y_indices = np.where(all_labels == y)[0].tolist()
                selected_idx = random.sample(all_y_indices, min(len(all_y_indices), args.max_memory))

                selected_ids = np.array(train_dataloader.dataset.get_input_ids())[selected_idx]
                selected_attn_masks = np.array(train_dataloader.dataset.get_attention_masks())[selected_idx]
                selected_labels = np.array(train_dataloader.dataset.get_labels())[selected_idx]
                selected_raw_texts = np.array(train_dataloader.dataset.get_raw_texts())[selected_idx]

                # convert the labels, aligned with lifelonglearning.utils_functions.MyDataset.__getitem__
                selected_labels = np.array(
                    [define_text_classification_label_orders(train_dataloader.dataset.dataset)[i-1]
                     for i in selected_labels])

                start = 0
                while start * TRAIN_BATCH_SIZE < len(selected_idx):
                    end = min((start + 1) * TRAIN_BATCH_SIZE, len(selected_idx))
                    tmp_ids = torch.tensor(selected_ids[start * TRAIN_BATCH_SIZE:end, :], device=device)
                    tmp_attn_masks = torch.tensor(selected_attn_masks[start * TRAIN_BATCH_SIZE:end, :], device=device)
                    tmp_raw_texts = selected_raw_texts[start * TRAIN_BATCH_SIZE:end]

                    tmp_keys = get_keys(key_encoder, tmp_ids, tmp_attn_masks)
                    memory.push(tmp_keys.cpu().numpy(),
                                (tmp_ids.cpu().numpy(), tmp_attn_masks.cpu().numpy(),
                                selected_labels[start * TRAIN_BATCH_SIZE:end], tmp_raw_texts),
                                args.max_memory, i,
                                criteria=MEM_GRP_CRTR_LBL, strategy=args.sample_strategy)
                    start += 1
                    del tmp_ids
                    del tmp_attn_masks
                    del tmp_raw_texts
                    del tmp_keys

                del selected_ids
                del selected_attn_masks
                del selected_labels
                del selected_raw_texts
            del _
            del train_dataloader
        except (NonMatchingSplitsSizesError or ExpectedMoreSplits or UnexpectedSplits) as e:
            print(e)
            print(f'============ Skip the configuration from {args.dataset_config[i]}. ============')
            # do not remove the configuration here in case we need to verify in main process again
            continue

    return memory


def save_memory_to_disk(args, memory):
    import csv
    save_dir = os.path.join(args.log_dir, MEMORY_FOLDER)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    groups = memory.group_indices_by_value(MEM_GRP_CRTR_CONFIG)    # group by config, {config:[list of keys]}

    for k,v in groups.items():
        save_config_dir = os.path.join(save_dir, args.dataset_config[k])
        Path(save_config_dir).mkdir(parents=True, exist_ok=True)

        memory_file = os.path.join(save_config_dir, MEMORY_FILE)
        with open(memory_file, 'w') as file:
            csv_writer = csv.writer(file)
            for instance_key in v:
                # the 2nd index is the label, the 4th index is the raw text
                csv_writer.writerow([str(memory.memory[instance_key][2]), memory.memory[instance_key][4]])


def main():
    args = set_model_parameters()
    device = CUDA if torch.cuda.is_available() else CPU

    # tokenizer and key encoder must be loaded from pretrained path
    tokenizer, key_encoder = load_tokenizer_keyencoder(args)
    key_encoder.to(device)

    memory = ReplayMemory()
    if args.start_idx > 0 and not args.no_replay:
        print(f'============ Reinitiate data from all tasks before {args.dataset_config[args.start_idx]}. ============')
        memory = reinitiate_memory_by_config(args, tokenizer, key_encoder=key_encoder, device=device)

    i = args.start_idx
    while i < args.end_idx:
        print(f'============ Start training data from {args.dataset_config[i]}. ============')
        config = args.dataset_config[i]

        try:
            train_dataloader, valid_dataloader, test_dataloader = \
                get_dataset_load_func(args.dataset)(args, config, tokenizer)
        except (NonMatchingSplitsSizesError or ExpectedMoreSplits or UnexpectedSplits) as e:
            # the exception are for [HuggingFace] Amazon Product Review dataset only
            print(e)
            print(f'============ Skip the configuration from {args.dataset_config[i]}. ============')
            args.dataset_config.remove(config)
            # do not update i here so that the loop will go to the next valid config
            continue

        model = load_model(args, config, num_labels=2)

        optimizer = AdamW(params=model.parameters(), lr=LEARNING_RATE, weight_decay=args.wd)
        model.to(device)

        for epoch in range(args.max_epoch):
            train(args, epoch, train_dataloader, valid_dataloader, test_dataloader,
                  device, model, tokenizer, optimizer, memory, i, key_encoder)
            print(f'Finish Epoch {epoch + 1} ..............................')

        del train_dataloader
        del valid_dataloader
        del test_dataloader
        del model
        del optimizer

        i += 1

    # save the memory (raw_text) to disk after all training
    save_memory_to_disk(args, memory)

    if args.print_stdout:
        sys.stdout.close()


if __name__ == "__main__":
    main()


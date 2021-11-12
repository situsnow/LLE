import argparse, sys, os
import subprocess
from tqdm import tqdm
import time
import copy
import torch
from sklearn.utils import check_random_state
from datasets.utils.info_utils import UnexpectedSplits, ExpectedMoreSplits, NonMatchingSplitsSizesError

from utils.constants import *
from utils.utils_functions import send_to_cuda, split_sentence, save_to_disk, join_sentence
from utils.utils_data import handle_task_order, get_dataset_categories, get_dataset_load_func, get_train_data, \
    load_amr_data_split, TEST_BATCH_SIZE, load_plain_memory, clean_data, count_eligible_examples
from utils.utils_model import load_tokenizer_keyencoder, load_model
from utils.utils_algo import get_lime_weights, get_lrp_weights


os.environ["TOKENIZERS_PARALLELISM"] = "false"
COLUMNS = {AMSRV:AMAZON_COLUMNS}


def set_model_parameters():
    parser = argparse.ArgumentParser(description='Collect teacher explanations for LLE')
    parser.add_argument('--data-dir', type=str, help="Directory to the data")
    parser.add_argument('--cache-dir', type=str, help="Directory to the cache data")
    parser.add_argument('--log-dir', type=str, default='bert_base_models',
                        help='the logging dir for the black-box model')

    # dataset
    parser.add_argument('--dataset', type=str, default=AMSRV)
    parser.add_argument('--split-ratio', nargs='+', default=[20000, 2000, 2000], help='the ratio for the dataset')
    parser.add_argument('--split-unit', type=str, default=HUGGINGFACE_SPLIT_ABS,
                        choices=[HUGGINGFACE_SPLIT_ABS, HUGGINGFACE_SPLIT_PERCENT])
    parser.add_argument('--start-idx', type=int, default=0, help='the start index of dataset config')
    parser.add_argument('--end-idx', type=int, default=-1, help='the end index of dataset config')
    parser.add_argument('--task-order', type=str, choices=[DIRECT, RANDOM, REVERSE],
                        default=DIRECT, help='the ordering of the task, need to save random')
    # blackbox model
    parser.add_argument('--arch', type=str, default=DISTILBERT, help='the architecture of the model')

    # collecting explanations from base
    parser.add_argument("--bs-original-explainer", type=str, default=LRP,
                        choices=[LIME, LRP])
    parser.add_argument('--bs-sample-size', type=int, default=None,
                        help='the size in generating perturbed neighbor samples')
    # for base explainer that requires sampling, such as lime/ kernel_shap/ deep_shap
    parser.add_argument('--bs-batch-size', type=int, help='the mini batch size to get the output from BERT.')
    parser.add_argument('--norm', type=str, default=None, help='the normalize method for base explanation.')
    parser.add_argument('--bs-max-length', type=int, default=300, help='the max length of example')
    parser.add_argument('--bs-collect-split', type=str, choices=[TRAIN, DEV, TEST, MEM, INF])
    parser.add_argument('--force-append', action='store_true',
                        help='the indication to force append explanations on existing files')

    # for replay memory
    parser.add_argument('--fs-train-src', type=str, default=FS_TRAIN_SRC,
                        help='the file name of the fairseq training src')
    parser.add_argument('--fs-train-tgt', type=str, default=FS_TRAIN_TGT,
                        help='the file name of the fairseq training tgt')
    parser.add_argument('--fs-dev-src', type=str, default=FS_DEV_SRC,
                        help='the file name of the fairseq validation src')
    parser.add_argument('--fs-dev-tgt', type=str, default=FS_DEV_TGT,
                        help='the file name of the fairseq validation tgt')
    parser.add_argument('--fs-test-src', type=str, default=FS_TEST_SRC,
                        help='the file name of the fairseq test src')
    parser.add_argument('--fs-test-tgt', type=str, default=FS_TEST_TGT,
                        help='the file name of the fairseq test tgt')
    parser.add_argument('--fs-mem-src', type=str, default=FS_MEM_SRC,
                        help='the file name of the fairseq memory src')
    parser.add_argument('--fs-mem-tgt', type=str, default=FS_MEM_TGT,
                        help='the file name of the fairseq memory tgt')
    parser.add_argument('--fs-inf-src', type=str, default=FS_INF_SRC,
                        help='the file name of the fairseq inference src')
    parser.add_argument('--fs-inf-tgt', type=str, default=FS_INF_TGT,
                        help='the file name of the fairseq inference tgt')

    # set default as sim to ensure heuristic and fairseq parameters align
    parser.add_argument('--am-heuristics', type=str, default=HEURISTICS_SIM,
                        choices=[HEURISTICS_RANDOM, HEURISTICS_SIM, HEURISTICS_NOMEM, HEURISTICS_OLDMEM])

    args, _ = parser.parse_known_args(sys.argv)

    args.data_splits = [TRAIN, DEV, TEST, MEM]
    args.log_splits = {TRAIN: (args.fs_train_src, args.fs_train_tgt),
                       DEV: (args.fs_dev_src, args.fs_dev_tgt), TEST: (args.fs_test_src, args.fs_test_tgt),
                       MEM: (args.fs_mem_src, args.fs_mem_tgt), INF: (args.fs_inf_src, args.fs_inf_tgt)}

    # for collecting explanation
    args.label_token = "lbl"
    args.label_format = "<{}-{}>"
    args.seed = 1234
    args.random_state = check_random_state(args.seed)
    args.categories = get_dataset_categories(args.dataset)

    # for amortized/explainer process
    args.pretrained_path = os.path.join(args.data_dir, args.dataset, "pretrained_" + args.arch)
    args.pretrained_encoder_path = os.path.join(args.data_dir, args.dataset, "pretrained_" + args.arch + "_encoder")

    args.dataset_config = handle_task_order(args, args.task_order)
    if args.end_idx == -1:
        args.end_idx = len(args.dataset_config)

    return args


def collect_inference_explanation(args, config_idx, tokenizer):
    # for inference: we do not need to collect the first and the current task
    # as the inference data already in their test files
    if config_idx == 0:
        raise RuntimeError(f'Skip the inference data collecting in task {args.dataset_config[config_idx]} '
                           f'as it is already saved in its test files: fs_test_src.exp and fs_test_tgt.exp.')

    config = args.dataset_config[config_idx]
    bbox_model = load_model(args, config, lrp=args.bs_original_explainer == LRP,
                            num_labels=33 if args.dataset == TEXT_CLASSIFICATION else 2)
    bbox_model.eval()
    bbox_model = send_to_cuda(bbox_model)

    prev_config_idx = config_idx - 1
    # will save all inference data in current config folder/fs_inf_src.exp and /fs_inf_tgt.exp
    src_file, tgt_file = join_path(args, config, args.bs_collect_split)
    while prev_config_idx >= 0:
        prev_task = args.dataset_config[prev_config_idx]
        # regenerate the teacher explanations for all previous tasks given current bbox of the test set

        train_data = None
        if args.bs_original_explainer in [KERNEL_SHAP, DEEP_SHAP]:
            train_dataloader, _, _ = get_dataset_load_func(args.dataset)(args, config, tokenizer)
            del _
            train_data = get_train_data(train_dataloader, args.bs_max_length)
        if args.dataset == AMSRV:
            inf_dataloader = load_amr_data_split(args, prev_task,
                                                 from_ratio=int(args.split_ratio[0]) + int(args.split_ratio[1]),
                                                 to_ratio=int(args.split_ratio[0]) + int(args.split_ratio[1]) +
                                                          int(args.split_ratio[2]),
                                                 batch_size=TEST_BATCH_SIZE, tokenizer=tokenizer)

        collect_teacher_explanations(args, inf_dataloader, tokenizer, bbox_model,
                                     src_file, tgt_file, train_split=False, train_data=train_data)
        print(f"========= Finish inference data: config = {config}, prev_config = {prev_task} =========", flush=True)

        prev_config_idx -= 1


def is_all_data_collected(args, task):
    for split in args.data_splits:
        src_path, tgt_path = join_path(args, task, split)
        if (not os.path.exists(src_path)) or (not os.path.exists(tgt_path)):
            return False    # collect
    return True    # train


def join_path(args, task, split):
    src, tgt = args.log_splits[split]
    src_path = os.path.join(args.explainer_log_dir, task, src)
    tgt_path = os.path.join(args.explainer_log_dir, task, tgt)
    return src_path, tgt_path


def collect_explanations(args, config_idx, tokenizer):
    config = args.dataset_config[config_idx]
    bbox_model = load_model(args, config, lrp=args.bs_original_explainer == LRP,
                            num_labels=33 if args.dataset == TEXT_CLASSIFICATION else 2)
    bbox_model.eval()
    bbox_model = send_to_cuda(bbox_model)

    dataloaders = {}
    train_data = None
    if args.bs_collect_split in [TRAIN, DEV, TEST] or args.bs_original_explainer == DEEP_SHAP:
        try:
            train_dataloader, dev_dataloader, test_dataloader = get_dataset_load_func(args.dataset)(args, config, tokenizer)
        except (NonMatchingSplitsSizesError or ExpectedMoreSplits or UnexpectedSplits) as e:
            print(e)
            print(f'============ Skip the configuration from {args.dataset_config[config_idx]}. ============')
            args.dataset_config.remove(config)
            # do not update config_idx here so that the loop will go to the next valid config
            return
        print(f"========= Finish loading data from {config} =========", flush=True)
        dataloaders = {TRAIN:train_dataloader, DEV:dev_dataloader, TEST:test_dataloader}

        # NOTE !! for K'SHAP, we do not require the train data any more
        # if args.bs_original_explainer == DEEP_SHAP:
        #     train_data = get_train_data(train_dataloader, args.bs_max_length)

    else:
        prev_config_idx = config_idx
        all_mem_paths = []
        while prev_config_idx >= 0:
            prev_task = args.dataset_config[prev_config_idx]
            # regenerate the teacher explanations for all previous tasks given current bbox
            # collect the memory instances from bert_base_models/memory/task/memory.txt -->[label:::raw_text]
            mem_data_path = os.path.join(args.log_dir, MEMORY_FOLDER, prev_task, MEMORY_FILE)
            all_mem_paths.append(mem_data_path)
            prev_config_idx -= 1
        # construct the data loader, the config here does not matter as we will not save into memory.csv
        dataloader = load_plain_memory(args, config, all_mem_paths, COLUMNS[args.dataset], tokenizer)
        dataloaders[MEM] = dataloader

    dataloader = dataloaders[args.bs_collect_split]

    src_file, tgt_file = join_path(args, config, args.bs_collect_split)
    if args.force_append or (not os.path.exists(src_file)) or (not os.path.exists(tgt_file)):
        collect_teacher_explanations(args, dataloader, tokenizer, bbox_model,
                                     src_file, tgt_file,
                                     train_split=True if args.bs_collect_split == TRAIN else False,
                                     train_data=train_data)
    else:
        print(f"========= Data file {src_file} and {tgt_file} already exist =========", flush=True)


def collect_teacher_explanations(args, dataloader, tokenizer, bbox_model,
                                 src_file, tgt_file, train_split=False, train_data=None):
    count = 0
    skip_count = 0
    total_lines = 0
    # if we force to append into the file, count the existing lines first
    pass_flag = True
    if args.force_append and os.path.exists(src_file):
        cmd = f'wc -l {src_file}'
        cmd_output = str(subprocess.check_output(cmd, shell=True))
        total_lines = int(cmd_output[2:cmd_output.find(' ')])
        pass_flag = False

    for step, batch in tqdm(enumerate(dataloader, 0)):
        batch_cp = copy.deepcopy(batch)
        del batch

        num_of_instances = len(batch_cp['raw_text'])
        X = [clean_data(batch_cp['raw_text'][ex]) for ex in range(num_of_instances)]

        if args.force_append and not pass_flag:
            # and (3 < len(split_sentence(X[ex])) < args.bs_max_length):
            skip_count += count_eligible_examples(X, args.bs_max_length)
            if total_lines > skip_count:
                continue
            else:
                pass_flag = True    # so that we don't need to come back to this if segment in the following batches

        # get black-box model prediction 'y_hat'
        with torch.no_grad():
            ids = send_to_cuda(batch_cp['input_ids'].clone().detach())
            attn_masks = send_to_cuda(batch_cp['attention_mask'].clone().detach())
            targets = send_to_cuda(batch_cp['labels'].clone().detach())
            output = bbox_model(ids, attn_masks)

            y_hats = torch.argmax(output[0], dim=1)

            del ids
            del attn_masks
        del batch_cp

        # if retrain_or_explain(args, config_idx, total) == RETRAIN:
        exp_list = []
        x_list = []
        for ex in range(num_of_instances):  # loop all examples in current batch
            # filter too long or too short examples
            if len(split_sentence(X[ex])) < 3 or len(split_sentence(X[ex])) > args.bs_max_length:
                continue
            # get explanation for current example
            args.explain_label = y_hats[ex].item()
            label_token = args.label_format.format(args.label_token, args.categories[y_hats[ex]])

            try:
                base_exp = get_base_explanation(args, count + ex, X[ex], targets[ex].item(),
                                                bbox_model, tokenizer, train_split=train_split, train_data=train_data)
            except RuntimeError:
                print('Cannot generate explanation for this example:')
                print(X[ex])
                continue

            if any(w != 0.0 for w in base_exp):
                # add triple(x, y_hat, r) to file
                exp_list.append(base_exp)
                x_list.append("%s %s %s" % (label_token, " ".join(X[ex]), label_token))

        if len(exp_list) == 0:
            continue

        print("========= Save current batch to disk =========", flush=True)
        save_to_disk(x_list, exp_list, src_file, tgt_file)

        count += num_of_instances


def get_base_explanation(am_args, x, bbox_model, tokenizer):

    start_time = time.time()
    base_exp = [0.0] * len(x)
    am_args.sample_size = am_args.bs_sample_size
    am_args.mini_batch = am_args.bs_batch_size
    if am_args.bs_original_explainer == LIME:
        with torch.no_grad():
            base_exp = get_lime_weights(am_args, x, bbox_model, tokenizer)
    elif am_args.bs_original_explainer == LRP:
        attention_mask = None
        input_id = send_to_cuda(torch.tensor([tokenizer.encode(join_sentence(x), truncation=True)]))
        y = bbox_model(input_id, attention_mask)[0]
        base_exp = get_lrp_weights(am_args, x, input_id, y, bbox_model, tokenizer)
        del input_id
        del y
    end_time = time.time()
    print(f'Total processing time for one instance:{str(end_time - start_time)}')
    return base_exp


def main():
    args = set_model_parameters()

    for config_idx in range(args.start_idx, args.end_idx):
        config = args.dataset_config[config_idx]

        print(f'================= Start New Explanation Process on {config} =================', flush=True)

        if args.bs_collect_split == INF:
            # tokenizer and key encoder must be loaded from pretrained pat
            tokenizer, _ = load_tokenizer_keyencoder(args)
            collect_inference_explanation(args, config_idx, tokenizer)
        elif args.force_append or (not is_all_data_collected(args, config)):
            tokenizer, _ = load_tokenizer_keyencoder(args)
            collect_explanations(args, config_idx, tokenizer)


if __name__ == "__main__":
    main()
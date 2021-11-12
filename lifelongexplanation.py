'''
The fairseq (https://github.com/pytorch/fairseq) version is a bit old
'''
import argparse
import sys, os
import torch


from utils.constants import *
from utils.utils_functions import copy_dict, remove_dict
from utils.utils_data import handle_task_order
from utils.utils_model import copy_checkpoint, remove_prev_checkpoint
from utils.utils_algo import read_fairseq_parameters

from fairseq.train_lle import cli_main


def set_model_parameters():
    parser = argparse.ArgumentParser(description='Life Long Explanation')
    parser.add_argument('--data-dir', type=str, default='/home/xsit1/snow/dataset/', help="Directory to the data")
    parser.add_argument('--cache-dir', type=str, help="Directory to the cache data")
    parser.add_argument('--log-dir', type=str, default='bert_base_models',
                        help='the logging dir for the black-box model')
    parser.add_argument('--explainer-log-dir', type=str, default=FAIRSEQ_FOLDER,
                        help="the logging dir for training the explainer")

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

    # explainer training
    parser.add_argument('--fs-explainer-type', type=str, default=MULTILABEL, choices=[RANK, MULTILABEL])
    parser.add_argument('--fs-explainer-parameters', type=str, default='sequential_explainer_parameters',
                        help='file name of the fairseq model parameters')
    parser.add_argument('--fs-print-file-log', action='store_true', help='indication to save log in file')
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

    # amortized process
    # set default as sim to ensure heuristic and fairseq parameters align
    parser.add_argument('--am-heuristics', type=str, default=HEURISTICS_RANDOM,
                        choices=[HEURISTICS_RANDOM, HEURISTICS_NOMEM, HEURISTICS_OLDMEM])

    parser.add_argument('--dict-file', type=str, default='dict.txt', help='the dict file for fairseq training')
    parser.add_argument('--new-dict-file', type=str, default='dict.txt',
                        help='the dict file for fairseq training')

    args, _ = parser.parse_known_args(sys.argv)

    args.data_splits = [TRAIN, DEV, TEST, MEM]    # DO NOT ADD INF HERE AS WE DO NOT NEED IT TO TRAIN EXPLAINER !!!

    args.log_splits = {TRAIN: (args.fs_train_src, args.fs_train_tgt),
                       DEV: (args.fs_dev_src, args.fs_dev_tgt), TEST: (args.fs_test_src, args.fs_test_tgt),
                       MEM: (args.fs_mem_src, args.fs_mem_tgt), INF: (args.fs_inf_src, args.fs_inf_tgt)}

    # for explainer process
    args.pretrained_path = os.path.join(args.data_dir, args.dataset, "pretrained_" + args.arch)
    args.pretrained_encoder_path = os.path.join(args.data_dir, args.dataset, "pretrained_" + args.arch + "_encoder")
    args.log_dir = os.path.join(args.data_dir, args.dataset, args.log_dir)

    args.explainer_log_dir = os.path.join(args.data_dir, args.dataset, args.explainer_log_dir,
                                          args.bs_original_explainer,
                                          'auto' if args.bs_original_explainer in [LRP, GRADIENT, OCCLUSION]
                                          else str(args.bs_sample_size))

    args.fs_explainer_parameters = "%s/%s" % (args.explainer_log_dir, args.fs_explainer_parameters)

    args.dataset_config = handle_task_order(args, args.task_order)
    if args.end_idx == -1:
        args.end_idx = len(args.dataset_config)

    return args


if __name__ == "__main__":
    # collect all hyper-parameters
    args = set_model_parameters()

    for config_idx in range(args.start_idx, args.end_idx):
        config = args.dataset_config[config_idx]

        print(f'================= Start New Explanation Process on {config} =================', flush=True)

        # train explainer
        # copy a historical memory/previous checkpoint to current folder
        if config_idx != 0:
            copy_checkpoint(args, config_idx)

        # copy dictionary and construct the system arguments before cli_main()
        copy_dict(args, config)
        read_fairseq_parameters(args, config_idx)

        cli_main()

        # free up the cuda memory
        torch.cuda.empty_cache()

        # remove the historical memory /previous checkpoint and dictionary in current folder
        if config_idx != 0:
            remove_prev_checkpoint(args, config_idx)
        remove_dict(args, config)
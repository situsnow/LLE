import argparse
import sys
import os
import time

from utils.constants import AMSRV, DIRECT, DISTILBERT, FS_DEV_SRC, FS_DEV_TGT, LRP, MULTILABEL, FAIRSEQ_RAN_PARAMETERS, \
    HEURISTICS_CHECKPOINT_DICT, HEURISTICS_RANDOM
from utils.utils_data import handle_task_order, load_src_tgt
from utils.utils_algo import prepare_load_fairseq_explainer, get_weights_from_explainer


def set_parameters():
    parser = argparse.ArgumentParser(description='Log odds Parser')
    parser.add_argument('--data-dir', type=str, default='/home/xsit1/snow/dataset/', help="Directory to the data")
    parser.add_argument('--explainer-log-dir', type=str, default='am_fs_random_wd0.1',
                        help="the logging dir for training the explainer")

    # dataset
    parser.add_argument('--dataset', type=str, default=AMSRV)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--task-order', type=str, default=DIRECT, help='the ordering of the task, need to save random')

    # blackbox model
    parser.add_argument('--arch', type=str, default=DISTILBERT, help='the architecture of the model')

    # explainer
    parser.add_argument('--fs-dev-src', type=str, default=FS_DEV_SRC,
                        help='the file name of the fairseq validation src')
    parser.add_argument('--fs-dev-tgt', type=str, default=FS_DEV_TGT,
                        help='the file name of the fairseq validation tgt')
    parser.add_argument('--original-explainer', type=str, default=LRP)
    parser.add_argument('--bs-sample-size', type=int, default=None,
                        help='the size in generating perturbed neighbor samples')
    parser.add_argument('--target-checkpoint', type=str, default='last',
                        choices=['last', 'best'])

    parser.add_argument('--explainer-type', type=str, default=MULTILABEL)

    args, _ = parser.parse_known_args(sys.argv)

    args.dataset_config = handle_task_order(args, args.task_order)
    args.explainer_log_dir = os.path.join(args.data_dir, args.dataset, args.explainer_log_dir,
                                          args.original_explainer, 'auto')
    args.fs_explainer_parameters = os.path.join(args.explainer_log_dir, FAIRSEQ_RAN_PARAMETERS)
    return args


if __name__ == "__main__":
    args = set_parameters()
    for t in range(len(args.dataset_config)):
        task = args.dataset_config[t]
        sq_explainer = prepare_load_fairseq_explainer(args, task,
                                                      HEURISTICS_CHECKPOINT_DICT[HEURISTICS_RANDOM],
                                                      args.fs_explainer_parameters,
                                                      retain_data_path=True,
                                                      target_checkpoint=args.target_checkpoint)

        file_path = os.path.join(args.explainer_log_dir, task)
        src, _ = load_src_tgt(os.path.join(file_path, args.fs_dev_src), os.path.join(file_path, args.fs_dev_tgt))
        for x in src:
            start_time = time.time()
            get_weights_from_explainer(sq_explainer, x, output_dim=3).squeeze(0).tolist()
            end_time = time.time()

            print(f'Total processing time for one instance:{str(end_time - start_time)}')
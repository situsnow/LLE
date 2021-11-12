import argparse, sys, os
import random
import numpy as np
import torch

from utils.utils_functions import send_to_cuda, join_sentence
from utils.constants import *

from utils.utils_data import handle_task_order, get_data_files, load_src_tgt
from utils.utils_functions import log_odds
from utils.utils_algo import masked_top_weight_tokens, get_weights_from_explainer, load_fairseq_args, \
    masked_random_tokens
from utils.utils_model import prediction_by_bert, load_fairseq_trainer, get_recent_checkpoint, \
    load_tokenizer_keyencoder, load_model

data_split = [TRAIN, DEV, TEST]


def set_parameters():
    parser = argparse.ArgumentParser(description='Log odds Parser')
    parser.add_argument('--data-dir', type=str, default='/home/xsit1/snow/dataset/', help="Directory to the data")
    parser.add_argument('--cache-dir', type=str, default='/home/xsit1/snow/cache/')
    parser.add_argument('--log-dir', type=str, default=BBOX_FOLDER,
                        help='the logging dir for the black-box model')
    parser.add_argument('--explainer-log-dir', type=str, default=FAIRSEQ_FOLDER,
                        help="the logging dir for training the explainer")

    # dataset
    parser.add_argument('--dataset', type=str, default=AMSRV)
    parser.add_argument('--doc-num', type=int, help='number of documents selected for evaluation')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--task-order', type=str, choices=[DIRECT, RANDOM, REVERSE],
                        default=DIRECT, help='the ordering of the task, need to save random')

    # blackbox model
    parser.add_argument('--arch', type=str, default=DISTILBERT, help='the architecture of the model')

    # explainer
    parser.add_argument('--fs-test-src', type=str, default=FS_TEST_SRC,
                        help='the file name of the fairseq test src')
    parser.add_argument('--fs-test-tgt', type=str, default=FS_TEST_TGT,
                        help='the file name of the fairseq test tgt')
    parser.add_argument('--fs-inf-src', type=str, default=FS_INF_SRC,
                        help='the file name of the fairseq inf src')
    parser.add_argument('--fs-inf-tgt', type=str, default=FS_INF_TGT,
                        help='the file name of the fairseq inf tgt')
    parser.add_argument('--original-explainer', type=str, default=LRP, choices=[LIME, LRP, GRADIENT, KERNEL_SHAP])
    parser.add_argument('--bs-sample-size', type=int, default=None,
                        help='the size in generating perturbed neighbor samples')
    parser.add_argument('--target-checkpoint', type=str, default='last',
                        choices=['last', 'best'])

    parser.add_argument('--cat', type=str, default=POS, choices=[POS, NEG], help='the category of weights')
    parser.add_argument('--explainer-type', type=str, default=MULTILABEL)
    parser.add_argument('--start-idx', type=int, default=0)
    parser.add_argument('--end-idx', type=int, default=-1)

    args, _ = parser.parse_known_args(sys.argv)

    random.seed(args.seed)

    args.dataset_config = handle_task_order(args, args.task_order)
    if args.end_idx == -1 or args.end_idx >= len(args.dataset_config):
        args.end_idx = len(args.dataset_config)

    args.dataset_path = os.path.join(args.data_dir, args.dataset)
    args.log_dir = os.path.join(args.dataset_path, args.log_dir)  # to load bbox
    # to load tokenizer
    args.pretrained_path = os.path.join(args.data_dir, args.dataset, "pretrained_" + args.arch)
    args.pretrained_encoder_path = os.path.join(args.data_dir, args.dataset, "pretrained_" + args.arch + "_encoder")

    args.explainer_log_dir = os.path.join(args.dataset_path, args.explainer_log_dir,
                                          args.original_explainer,
                                          'auto' if args.original_explainer in [LRP, GRADIENT, OCCLUSION]
                                          else str(args.bs_sample_size))
    args.sq_explainer_parameters = os.path.join(args.explainer_log_dir, FAIRSEQ_RAN_PARAMETERS)
    args.sq_explainer_parameters_oldmem = os.path.join(args.explainer_log_dir, FAIRSEQ_OLDMEM_PARAMETERS)
    args.sq_explainer_parameters_nomem = os.path.join(args.explainer_log_dir, FAIRSEQ_NOMEM_PARAMETERS)

    return args


def masked_explainer_tokens(src, weights, masked_token, args, interval, lrp=False):
    # multilabel, not lrp
    if lrp:
        # multilabel, lrp
        # 0: high pos; 1: medium pos; 2: low pos
        high_weight = [i for i in range(1, len(src) - 1) if weights[i - 1] == 0]
        medium_weight = [i for i in range(1, len(src) - 1) if weights[i - 1] == 1]

        candidates = []
        if interval <= len(high_weight):
            candidate_weights = high_weight
        elif interval <= len(high_weight) + len(medium_weight):
            candidates = high_weight  # we will select all high_weight and sample the rest from medium_weight
            candidate_weights = medium_weight
        else:
            # we will select all high and medium weight, and sample the rest from low_weight
            # candidates = high_weight + medium_weight
            # candidate_weights = [i for i in range(1, len(src) - 1) if weights[i - 1] == 2]
            # since we only consider high + med pos in interval selection, should not sample from low weight here
            candidate_weights = high_weight + medium_weight
        candidates += random.sample(candidate_weights, min(interval - len(candidates), len(candidate_weights)))

    elif args.cat.lower() == POS:
        # 0: pos
        pos_weight = [i for i in range(1, len(src) - 1) if weights[i - 1] == 0]
        # medium_weight = list(filter(None, [i if weights[i-1] == 3 else None for i in range(1, len(src) - 1)]))
        candidates = random.sample(pos_weight, min(interval, len(pos_weight)))
    else:
        # 2: neg;
        neg_weights = [i for i in range(1, len(src) - 1) if weights[i - 1] == 2]
        candidates = random.sample(neg_weights, min(interval, len(neg_weights)))
        # Note !! In neg, we do not sample from all neg indices in case any explainer gets biased by pos.
        # candidates = [i for i in range(1, len(src) - 1) if weights[i - 1] == 2]
    new_src = [masked_token if i in candidates else src[i] for i in range(1, len(src) - 1)]

    return new_src


def get_interval_by_type(args, explainer_tgt, lrp=False):
    if not lrp:
        if args.cat == 'pos':
            # 0: pos
            interval = sum([1 if x == 0 else 0 for x in explainer_tgt])
        else:
            # 2: neg;
            interval = sum([1 if x == 2 else 0 for x in explainer_tgt])
    else:
        # lrp, there could only be args.cat == pos
        # 0: high pos; 1: medium pos; 2: low pos
        interval = sum([1 if x == 0 or x == 1 else 0 for x in explainer_tgt])
        if interval == len(explainer_tgt):
            interval = sum([1 if x == 0 else 0 for x in explainer_tgt])

    return interval


def print_dict(logodds_dict):
    for k, v in logodds_dict.items():
        print(f"{k}:{v}")


def calculate_logodds(args, tokenizer, bbox_model, interval, src=None, y_hat=None, explainer=None,
                      teacher=False, tgt=None, masked_src=None):
    if teacher and tgt is None:
        raise RuntimeError("The target should not be empty when it is teacher explanation.")

    with torch.no_grad():
        if masked_src is None and teacher:
            masked_src = masked_top_weight_tokens(args, src, tgt, tokenizer.mask_token, interval=interval)
        elif masked_src is None:
            # LLE
            tgt = get_weights_from_explainer(explainer, src, output_dim=3).squeeze(0).tolist()
            masked_src = masked_explainer_tokens(src, tgt, tokenizer.mask_token, args,
                                                 interval, lrp=args.original_explainer == LRP)

        # masked_src should be passed in for random cases
        masked_pred = prediction_by_bert(tokenizer, bbox_model, join_sentence(masked_src))

    lo = log_odds(masked_pred[y_hat])
    return lo


def prepare_load_fairseq_explainer(args, task, explainer_name, explainer_parameters, retain_data_path=False,
                                   target_checkpoint='last'):
    explainer_log_dir = os.path.join(args.explainer_log_dir, task, explainer_name)
    fairseq_args = load_fairseq_args(explainer_parameters)
    if not retain_data_path:
        fairseq_args.data = os.path.join(fairseq_args.data, task)
    explainer = load_fairseq_trainer(fairseq_args,
                                     get_recent_checkpoint(explainer_log_dir, target_checkpoint=target_checkpoint))
    return explainer


def main():
    # aggregate per task i
    agg_logodds_lle, agg_logodds_base, agg_logodds_random = [], [], []
    agg_logodds_lle_oldmem, agg_logodds_lle_nomem = [], []

    args = set_parameters()

    bbox_model, sq_explainer = None, None
    tokenizer, _ = load_tokenizer_keyencoder(args)
    del _

    for t in range(args.start_idx, args.end_idx):
        cur_config = args.dataset_config[t]

        # in case there are too many copies
        del bbox_model
        # load the black box of current task t
        bbox_model = load_model(args, cur_config, lrp=args.original_explainer == LRP,
                                num_labels=2)
        bbox_model = send_to_cuda(bbox_model)

        # in case there are too many copies
        del sq_explainer
        # load LLE explainer (sequential) checkpoint at time t
        sq_explainer = prepare_load_fairseq_explainer(args, cur_config,
                                                      HEURISTICS_CHECKPOINT_DICT[HEURISTICS_RANDOM],
                                                      args.sq_explainer_parameters,
                                                      retain_data_path=True,
                                                      target_checkpoint=args.target_checkpoint)  # get dict.txt from upper level
        sq_explainer_oldmem = prepare_load_fairseq_explainer(args, cur_config,
                                                             HEURISTICS_CHECKPOINT_DICT[HEURISTICS_OLDMEM],
                                                             args.sq_explainer_parameters_oldmem,
                                                             retain_data_path=True,
                                                             target_checkpoint=args.target_checkpoint)
        sq_explainer_nomem = prepare_load_fairseq_explainer(args, cur_config,
                                                            HEURISTICS_CHECKPOINT_DICT[HEURISTICS_NOMEM],
                                                            args.sq_explainer_parameters_nomem,
                                                            retain_data_path=True,
                                                            target_checkpoint=args.target_checkpoint)

        # collect all delta logodds for current task t and all previous tasks from the same file
        logodds_lle_t, logodds_base_t, logodds_random_t = [], [], []
        logodds_lle_oldmem_t, logodds_lle_nomem_t = [], []

        count = 0
        splits = [INF, TEST] if t > 0 else [TEST]
        for split in splits:
            args.data_files = get_data_files(args, cur_config)
            (src_path, tgt_path) = args.data_files[split]
            src, tgt = load_src_tgt(src_path, tgt_path)

            for i in range(len(src)):
                x = src[i]
                y = tgt[i]

                # get the top k important words acording to the explainer
                with torch.no_grad():
                    # we only categorize into 0: pos, 1: neu, 2: neg for LIME
                    # and 0: high pos; 1: medium pos; 2: low pos for LRP
                    explainer_tgt = get_weights_from_explainer(sq_explainer, x,
                                                               output_dim=3).squeeze(0).tolist()
                interval = get_interval_by_type(args, explainer_tgt, lrp=args.original_explainer == LRP)

                if interval == 0 or interval == len(x):
                    print(f"Skip {cur_config}-{split}-{i} "
                          f"because LLE is not able to generate good explanations.")
                    continue

                # get original prediction
                pred = prediction_by_bert(tokenizer, bbox_model, join_sentence(x[1:-1]))
                y_hat = np.argmax(pred)

                # ===============
                # logodds per doc
                logodds_original = log_odds(pred[y_hat])

                # delta logodds per doc in the eval task j
                logodds_lle = calculate_logodds(args, tokenizer, bbox_model, interval, x, y_hat, sq_explainer)
                logodds_lle_t.append(logodds_original - logodds_lle)

                logodds_lle_oldmem = calculate_logodds(args, tokenizer, bbox_model, interval, x, y_hat,
                                                       sq_explainer_oldmem)
                logodds_lle_oldmem_t.append(logodds_original - logodds_lle_oldmem)

                logodds_lle_nomem = calculate_logodds(args, tokenizer, bbox_model, interval, x, y_hat,
                                                      sq_explainer_nomem)
                logodds_lle_nomem_t.append(logodds_original - logodds_lle_nomem)

                logodds_base = calculate_logodds(args, tokenizer, bbox_model, interval, x,
                                                 y_hat, teacher=True, tgt=y)
                logodds_base_t.append(logodds_original - logodds_base)

                # random
                random_masked_srcs = masked_random_tokens(x, interval, tokenizer.mask_token)
                random_logodds = []
                for masked_src in random_masked_srcs:
                    logodds_random = calculate_logodds(args, tokenizer, bbox_model, interval, y_hat=y_hat,
                                                       masked_src=masked_src)
                    random_logodds.append(logodds_original - logodds_random)
                logodds_random_t.append(min(random_logodds) if args.cat == POS else max(random_logodds))

            print(f'============ All Base logodds in {split}:')
            print(logodds_base_t[count:])
            print(f'============ All LLE logodds in {split}:')
            print(logodds_lle_t[count:])
            print(f'============ All LLE old memory logodds in {split}:')
            print(logodds_lle_oldmem_t[count:])
            print(f'============ All LLE no memory logodds in {split}:')
            print(logodds_lle_nomem_t[count:])
            print(f'============ All Random logodds in {split}:')
            print(logodds_random_t[count:])

            count = len(logodds_lle_t)

        agg_logodds_lle.append(sum(logodds_lle_t) / len(logodds_lle_t))
        agg_logodds_lle_oldmem.append(sum(logodds_lle_oldmem_t) / len(logodds_lle_oldmem_t))
        agg_logodds_lle_nomem.append(sum(logodds_lle_nomem_t) / len(logodds_lle_nomem_t))
        agg_logodds_base.append(sum(logodds_base_t) / len(logodds_base_t))
        agg_logodds_random.append(sum(logodds_random_t) / len(logodds_random_t))

    print(f'########### Final Base logodds:')
    print(agg_logodds_base)
    print(f'########### Final LLE logodds:')
    print(agg_logodds_lle)
    print(f'########### Final LLE old memory logodds:')
    print(agg_logodds_lle_oldmem)
    print(f'########### Final LLE no memory logodds:')
    print(agg_logodds_lle_nomem)
    print(f'########### Final Random logodds:')
    print(agg_logodds_random)


if __name__ == "__main__":
    main()





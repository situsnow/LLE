import argparse
import os.path
import sys
import copy
import random
import torch.cuda
import numpy as np

from utils.utils_functions import read_plain_file

from utils.constants import *

from utils.utils_data import handle_task_order, load_src_tgt, get_data_files
from utils.utils_model import convert_to_bert_hidden_states, convert_to_n_gram, bert_cos_sim, n_gram_sim
from utils.utils_algo import get_weights_from_explainer, prepare_load_fairseq_explainer


delimiter = ':::'
OUR_EXP_POS = "our_exp_pos"
OUR_EXP_NEG = "our_exp_neg"
OUR_EXP_OLDMEM_POS = "our_exp_oldmem_pos"
OUR_EXP_OLDMEM_NEG = "our_exp_oldmem_neg"
OUR_EXP_NOMEM_POS = "our_exp_nomem_pos"
OUR_EXP_NOMEM_NEG = "our_exp_nomem_neg"
BASE_EXP_POS = "base_exp_pos"
BASE_EXP_NEG = "base_exp_neg"
RANDOM_EXP_POS = "random_exp_pos"
RANDOM_EXP_NEG = "random_exp_neg"


def set_parameters():
    parser = argparse.ArgumentParser(description='Find neighbours or measure IoU parser')
    parser.add_argument('--data-dir', type=str, default='/home/xsit1/snow/dataset/', help="Directory to the data")
    parser.add_argument('--explainer-log-dir', type=str, default=FAIRSEQ_FOLDER,
                        help="the logging dir for training the explainer")
    parser.add_argument('--bert-dir', type=str, default=PRETRAINED_BERT_FOLDER,
                        help='the dir to semantic similarity measure (BERT)')

    # measure similarity or IoU
    parser.add_argument('--pgm-option', type=str, default=PGM_OPTION_SIMILARITY,
                        choices=[PGM_OPTION_SIMILARITY, PGM_OPTION_IOU],
                        help='the option to collect similarities or to measure stability (iou).')
    parser.add_argument('--len-interval', type=int, default=20,
                        help='the length interval to group examples')  # e.g., 1-20, 21-40, 41-60, ...
    parser.add_argument('--sim-metric', type=str, default=N_GRAM, choices=[BERT_COS, N_GRAM])
    parser.add_argument('--nn', type=int, default=3, help='number of nearest neighbours')
    parser.add_argument('--log-precision', default=False, action='store_true',
                        help='if logarithm is required in similarity metric, only availble in n-gram, '
                             'default No logarithm.')

    # dataset
    parser.add_argument('--dataset', type=str, default=AMSRV)
    parser.add_argument('--doc-num', type=int, help='number of documents selected for evaluation')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--task-order', type=str, choices=[DIRECT, RANDOM, REVERSE],
                        default=DIRECT, help='the ordering of the task, need to save random')

    # explainer
    parser.add_argument('--fs-test-src', type=str, default=FS_TEST_SRC,
                        help='the file name of the fairseq test src')
    parser.add_argument('--fs-test-tgt', type=str, default=FS_TEST_TGT,
                        help='the file name of the fairseq test tgt')
    parser.add_argument('--fs-inf-src', type=str, default=FS_INF_SRC,
                        help='the file name of the fairseq inf src')
    parser.add_argument('--fs-inf-tgt', type=str, default=FS_INF_TGT,
                        help='the file name of the fairseq inf tgt')
    parser.add_argument('--original-explainer', type=str, default=LRP,
                        choices=[LIME, LRP])
    parser.add_argument('--bs-sample-size', type=int, default=None,
                        help='the size in generating perturbed neighbor samples')
    parser.add_argument('--target-checkpoint', type=str, default='last',
                        choices=['last', 'best'])

    # logging
    parser.add_argument('--print-examples', action='store_false',
                        help='logging the similar words in neighbouring examples, default True')

    args, _ = parser.parse_known_args(sys.argv)

    # just to set a default value for reusing previous pgm
    args.categorize_type = MULTILABEL

    args.bert_dir = os.path.join(args.data_dir, args.bert_dir)

    args.explainer_log_dir = os.path.join(args.data_dir, args.dataset, args.explainer_log_dir, args.original_explainer,
                                          'auto' if args.original_explainer in [LRP, GRADIENT, OCCLUSION]
                                          else str(args.bs_sample_size))
    args.sq_explainer_parameters = os.path.join(args.explainer_log_dir, FAIRSEQ_RAN_PARAMETERS)
    args.sq_explainer_parameters_oldmem = os.path.join(args.explainer_log_dir, FAIRSEQ_OLDMEM_PARAMETERS)
    args.sq_explainer_parameters_nomem = os.path.join(args.explainer_log_dir, FAIRSEQ_NOMEM_PARAMETERS)

    # e.g., similarity_inf_bert.txt
    args.similarity_file = {INF: 'similarity_inf_%s.txt' % args.sim_metric,
                            TEST: 'similarity_test_%s.txt' % args.sim_metric}

    args.dataset_config = handle_task_order(args, args.task_order)

    return args


def select_from_lrp_explainer(weight, interval, src):
    high_pos_idx = [i for i in range(len(weight)) if weight[i] == 0]
    med_pos_idx = [i for i in range(len(weight)) if weight[i] == 1]
    candidates = []
    if interval <= len(high_pos_idx):
        candidate_weights = high_pos_idx
    elif interval <= len(high_pos_idx) + len(med_pos_idx):
        candidates = high_pos_idx
        candidate_weights = med_pos_idx
    else:
        # candidates = high_pos_idx + med_pos_idx
        # candidate_weights = [i for i in range(len(weight)) if weight[i] == 2]
        # since we only consider high + med pos in interval, should not sample from low weight here
        candidate_weights = high_pos_idx + med_pos_idx
    candidates += random.sample(candidate_weights, min(interval - len(candidates), len(candidate_weights)))
    return [src[i + 1] for i in sorted(candidates)]


def get_exp_tuple(src, base_weight, explainer, explainer_oldmem, explainer_nomem, lrp=False):

    our_weight = get_weights_from_explainer(explainer, src, output_dim=3).squeeze(0).tolist()
    our_oldmem_weight = get_weights_from_explainer(explainer_oldmem, src, output_dim=3).squeeze(0).tolist()
    our_nomem_weight = get_weights_from_explainer(explainer_nomem, src, output_dim=3).squeeze(0).tolist()

    def sort_base_weight(i):
        return base_weight[i]
    order = sorted(range(len(base_weight)), key=lambda i: sort_base_weight(i), reverse=True)

    our_exp_pos, our_exp_neg, base_exp_pos, base_exp_neg, random_exp_pos, random_exp_neg = \
        [], [], [], [], [], []
    our_exp_oldmem_pos, our_exp_oldmem_neg, our_exp_nomem_pos, our_exp_nomem_neg = \
        [], [], [], []

    # multilabel - non lrp
    if not lrp:
        # 0: pos; 1: neutral; 2: neg
        pos_interval = sum([1 if x == 0 else 0 for x in our_weight])
        neg_interval = sum([1 if x == 2 else 0 for x in our_weight])

        our_exp_pos = list(filter(None, [src[i + 1] if our_weight[i] == 0 else None
                                         for i in range(len(our_weight))]))
        our_exp_neg = list(filter(None, [src[i + 1] if our_weight[i] == 2 else None
                                         for i in range(len(our_weight))]))

        if pos_interval != 0:
            # pos_idx in oldmem
            pos_idx_oldmem = [i for i in range(len(our_oldmem_weight)) if our_oldmem_weight[i] == 0]
            selected_oldmem_pos = random.sample(pos_idx_oldmem, min(pos_interval, len(pos_idx_oldmem)))
            our_exp_oldmem_pos = [src[i + 1] for i in sorted(selected_oldmem_pos)]

            # pos_idx in nomem
            pos_idx_nomem = [i for i in range(len(our_nomem_weight)) if our_nomem_weight[i] == 0]
            selected_nomem_pos = random.sample(pos_idx_nomem, min(pos_interval, len(pos_idx_nomem)))
            our_exp_nomem_pos = [src[i + 1] for i in sorted(selected_nomem_pos)]

            base_exp_pos = list(filter(None, [src[i + 1] if i in order[:pos_interval] and base_weight[i] > 0.0 else None
                                              for i in range(len(base_weight))]))
            pos_idx = random.sample(range(1, len(src)-1), pos_interval)   # already filter the front/end label tag
            random_exp_pos = [src[i] for i in sorted(pos_idx)]

        if neg_interval != 0:
            # neg_idx in oldmem
            neg_idx_oldmem = [i for i in range(len(our_oldmem_weight)) if our_oldmem_weight[i] == 2]
            selected_oldmem_neg = random.sample(neg_idx_oldmem, min(neg_interval, len(neg_idx_oldmem)))
            our_exp_oldmem_neg = [src[i + 1] for i in sorted(selected_oldmem_neg)]

            # neg_idx in nomem
            neg_idx_nomem = [i for i in range(len(our_nomem_weight)) if our_nomem_weight[i] == 2]
            selected_nomem_neg = random.sample(neg_idx_nomem, min(neg_interval, len(neg_idx_nomem)))
            our_exp_nomem_neg = [src[i + 1] for i in sorted(selected_nomem_neg)]

            base_exp_neg = list(filter(None, [src[i + 1] if i in order[-neg_interval:] and base_weight[i] < 0.0
                                              else None for i in range(len(base_weight))]))
            neg_idx = random.sample(range(1, len(src)-1), neg_interval)
            random_exp_neg = [src[i] for i in sorted(neg_idx)]   # already filter the front/end label tag
    else:
        # multilabel - lrp (only pos)
        # 0: high pos; 1: medium pos; 2: low pos

        pos_interval = sum([1 if x == 0 or x == 1 else 0 for x in our_weight])
        our_exp_pos = list(filter(None, [src[i + 1] if our_weight[i] == 0 or our_weight[i] == 1 else None
                                         for i in range(len(our_weight))]))

        if pos_interval != 0:
            # oldmem
            our_exp_oldmem_pos = select_from_lrp_explainer(our_oldmem_weight, pos_interval, src)
            # nomem
            our_exp_nomem_pos = select_from_lrp_explainer(our_nomem_weight, pos_interval, src)

            mean_base_weight = np.mean(base_weight)
            std_base_weight = np.std(base_weight)
            base_exp_pos = list(
                filter(None, [src[i + 1] if i in order[:pos_interval]
                              and base_weight[i] > mean_base_weight - std_base_weight
                              else None for i in range(len(base_weight))]))
            pos_idx = random.sample(range(1, len(src) - 1), pos_interval)
            random_exp_pos = [src[i] for i in sorted(pos_idx)]   # already filter the front/end label tag

    compare_tuple = {OUR_EXP_POS: our_exp_pos, OUR_EXP_NEG: our_exp_neg,
                     OUR_EXP_OLDMEM_POS: our_exp_oldmem_pos, OUR_EXP_OLDMEM_NEG: our_exp_oldmem_neg,
                     OUR_EXP_NOMEM_POS: our_exp_nomem_pos, OUR_EXP_NOMEM_NEG: our_exp_nomem_neg,
                     BASE_EXP_POS: base_exp_pos, BASE_EXP_NEG: base_exp_neg,
                     RANDOM_EXP_POS: random_exp_pos, RANDOM_EXP_NEG: random_exp_neg}
    return compare_tuple


def group_src(args, src):

    label_src = {}
    len_src = {}

    def get_len_from_src(i):
        return len(src[i]) - 2    # exclude the front/back tag

    # x[0][5:-1], <lbl-Business> --> Business
    for i in range(len(src)):
        length = get_len_from_src(i)
        len_interval = length // args.len_interval

        label_src[i] = src[i][0][5:-1]

        try:
            len_src[len_interval].append(i)
        except KeyError:
            len_src[len_interval] = [i]

    return label_src, len_src


def append_to_file(file_name, string):
    with open(file_name, 'a') as f:
        f.write(string + '\n')
        f.flush()


def find_previous_available_key(len_group, k):
    while k > 0:
        if (k - 1) in len_group:
            return k - 1
        else:
            k -= 1
    return k


def special_len_group(len_group):
    new_len_group = {}    # error will be thrown if the sorted_len_group changes size during iteration
    sorted_len_group = dict(sorted(len_group.items()))
    for k, v in sorted_len_group.items():
        if len(v) < 10 and k != 0:
            # merge to the previous available key
            prev_k = find_previous_available_key(new_len_group, k)
            new_len_group[prev_k].extend(v)
        else:
            new_len_group[k] = v

    return new_len_group


def collect_neighbours(args, task, split, bert_tokenizer, pretrained_bert, src):
    # group the src with format {label:src_idx} and {len interval: src_idx}, interval could be 50
    label_group, len_group = group_src(args, src)
    len_group = special_len_group(len_group)

    convert_fn = convert_to_bert_hidden_states if args.sim_metric == BERT_COS else convert_to_n_gram
    sim_metric = bert_cos_sim if args.sim_metric == BERT_COS else n_gram_sim
    save_file = os.path.join(args.explainer_log_dir, task, args.similarity_file[split])

    def measure_pairwise_distance(converted_x, neighbour):
        converted_nn = convert_fn(bert_tokenizer, pretrained_bert, neighbour)
        return sim_metric(converted_x, converted_nn, args.log_precision)

    # loop all instances by their predicted label
    for x_idx,label in label_group.items():

        converted_x = convert_fn(bert_tokenizer, pretrained_bert, src[x_idx])
        # find the len interval according its length
        len_interval = (len(src[x_idx]) - 2) // args.len_interval
        # find the candidates neighbors in group[len interval] that has the same label
        try:
            # needs a new copy as remove x_idx will remove in original copy
            neighbours = copy.deepcopy(len_group[len_interval])
        except KeyError:
            new_len_intveral = find_previous_available_key(len_group, len_interval)
            neighbours = copy.deepcopy(len_group[new_len_intveral])
        neighbours.remove(x_idx)
        # filter neighbours not has same label as current x_idx
        neighbours = [nn for nn in neighbours if label_group[nn] == label]
        if len(neighbours) == 0:
            print(f'Cannot find any neighbours in {task}:{split}:{x_idx}.')
        else:
            # measure the pairwise similarity based on metric [semantic/syntactic]
            distance = [measure_pairwise_distance(converted_x, src[nn]) for nn in neighbours]
            # sort the similarity and save into file with format [x idx: sorted neighbor indices]
            sorted_neighbours = [nn for _, nn in sorted(zip(distance,neighbours), reverse=True)]
            # only save the top-10 nearest neighbors
            min_num = min(10, len(sorted_neighbours))
            append_to_file(save_file, str(x_idx) + delimiter +
                           ','.join([str(nn) for nn in sorted_neighbours[:min_num]]))


def read_similarity_file(file_name):
    lines = read_plain_file(file_name)

    neighbour_dict = {}
    for line in lines:
        line = line.strip().split(delimiter)
        key = int(line[0])
        values = list(filter(None, line[1].strip().split(',')))
        neighbour_dict[key] = [int(v) for v in values] if len(values) > 0 else []

    return neighbour_dict


def print_logs(doc, compare_tuple, neighbour=False, lrp=False):
    if not neighbour:
        print('##### Current document:')
    else:
        print('##### Neighbor documents:')
    print(' '.join(doc))

    print('##### Our Explanation (Pos):')
    print(" ".join(compare_tuple[OUR_EXP_POS]))

    print('##### Our Explanation Old Memory (Pos):')
    print(" ".join(compare_tuple[OUR_EXP_OLDMEM_POS]))

    print('##### Our Explanation No Memory (Pos):')
    print(" ".join(compare_tuple[OUR_EXP_NOMEM_POS]))

    print('##### Base Explanation (Pos):')
    print(" ".join(compare_tuple[BASE_EXP_POS]))

    print('##### Random Explanation (Pos):')
    print(" ".join(compare_tuple[RANDOM_EXP_POS]))

    if not lrp:
        print('##### Our Explanation (Neg):')
        print(" ".join(compare_tuple[OUR_EXP_NEG]))

        print('##### Our Explanation Old Memory (Neg):')
        print(" ".join(compare_tuple[OUR_EXP_OLDMEM_NEG]))

        print('##### Our Explanation No Memory (Neg):')
        print(" ".join(compare_tuple[OUR_EXP_NOMEM_NEG]))

        print('##### Base Explanation (Neg):')
        print(" ".join(compare_tuple[BASE_EXP_NEG]))

        print('##### Random Explanation (Neg):')
        print(" ".join(compare_tuple[RANDOM_EXP_NEG]))


def measure_iou(current_compare_tuple, neighbour_compare_tuple, pos_key, neg_key):
    pos_union = list(set(current_compare_tuple[pos_key] + neighbour_compare_tuple[pos_key]))
    neg_union = list(set(current_compare_tuple[neg_key] + neighbour_compare_tuple[neg_key]))

    pos_intersect = list(set(current_compare_tuple[pos_key]) & set(neighbour_compare_tuple[pos_key]))
    neg_intersect = list(set(current_compare_tuple[neg_key]) & set(neighbour_compare_tuple[neg_key]))

    # IoU nuerator
    numerator = len(pos_intersect) + len(neg_intersect)

    # IoU denominator
    denominator = len(pos_union) + len(neg_union)

    return 0.0 if denominator == 0 else numerator/denominator


def measure_stability(args, task, split, neighbour_dict, src, tgt):
    # load the lle explainer
    sq_explainer = prepare_load_fairseq_explainer(args, task,
                                                  HEURISTICS_CHECKPOINT_DICT[HEURISTICS_RANDOM],
                                                  args.sq_explainer_parameters,
                                                  retain_data_path=True,
                                                  target_checkpoint=args.target_checkpoint)  # get dict.txt from upper level
    sq_explainer_oldmem = prepare_load_fairseq_explainer(args, task,
                                                         HEURISTICS_CHECKPOINT_DICT[HEURISTICS_OLDMEM],
                                                         args.sq_explainer_parameters_oldmem,
                                                         retain_data_path=True, target_checkpoint=args.target_checkpoint)
    sq_explainer_nomem = prepare_load_fairseq_explainer(args, task,
                                                        HEURISTICS_CHECKPOINT_DICT[HEURISTICS_NOMEM],
                                                        args.sq_explainer_parameters_nomem,
                                                        retain_data_path=True, target_checkpoint=args.target_checkpoint)

    lle_iou_split, base_iou_split, random_iou_split = [], [], []
    lle_iou_oldmem_split, lle_iou_nomem_split = [], []
    for i in range(len(src)):
        try:
            # find the nearest neighbors in similarity_dict
            neighbours = neighbour_dict[i][:min(args.nn, len(neighbour_dict[i]))]
        except KeyError:
            print(f'There is no neighbour found for instance {task}-{split}-{i}.')
            continue
        current_compare_tuple = get_exp_tuple(src[i], tgt[i],
                                              sq_explainer, sq_explainer_oldmem, sq_explainer_nomem,
                                              lrp=args.original_explainer == LRP)
        if args.print_examples:
            print('-------------------------------------------------------------------')
            print_logs(src[i], current_compare_tuple, lrp=args.original_explainer == LRP)

        lle_iou, base_iou, random_iou = 0.0, 0.0, 0.0
        lle_oldmem_iou, lle_nomem_iou = 0.0, 0.0

        if len(neighbours) > 0:
            for nn in neighbours:
                neighbour_compare_tuple = get_exp_tuple(src[nn], tgt[nn],
                                                        sq_explainer, sq_explainer_oldmem, sq_explainer_nomem,
                                                        lrp=args.original_explainer == LRP)
                if args.print_examples:
                    print('------------------')
                    print_logs(src[nn], neighbour_compare_tuple, neighbour=True, lrp=args.original_explainer == LRP)

                # measure IoU
                lle_iou += measure_iou(current_compare_tuple, neighbour_compare_tuple, OUR_EXP_POS, OUR_EXP_NEG)
                lle_oldmem_iou += measure_iou(current_compare_tuple, neighbour_compare_tuple,
                                              OUR_EXP_OLDMEM_POS, OUR_EXP_OLDMEM_NEG)
                lle_nomem_iou += measure_iou(current_compare_tuple, neighbour_compare_tuple,
                                             OUR_EXP_NOMEM_POS, OUR_EXP_NOMEM_NEG)

                base_iou += measure_iou(current_compare_tuple, neighbour_compare_tuple, BASE_EXP_POS, BASE_EXP_NEG)
                random_iou += measure_iou(current_compare_tuple, neighbour_compare_tuple,
                                          RANDOM_EXP_POS, RANDOM_EXP_NEG)

            avg_random_iou = random_iou / len(neighbours) * 100
            avg_base_iou = base_iou / len(neighbours) * 100
            avg_lle_iou = lle_iou / len(neighbours) * 100
            avg_lle_oldmem_iou = lle_oldmem_iou / len(neighbours) * 100
            avg_lle_nomem_iou = lle_nomem_iou / len(neighbours) * 100

            if args.print_examples:
                print(f'In {i}-th instance of {task}:{split} split:', end='')
                print('the average stability of random explanation is %.2f%%, the base explanation is %.2f%%, '
                      'lle explanation is %.2f%%, '
                      'lle old memory explanation is %.2f%%, lle no memory explanation is %.2f%%.'
                      % (avg_random_iou, avg_base_iou, avg_lle_iou, avg_lle_oldmem_iou, avg_lle_nomem_iou))
            lle_iou_split.append(avg_lle_iou)
            lle_iou_oldmem_split.append(avg_lle_oldmem_iou)
            lle_iou_nomem_split.append(avg_lle_nomem_iou)
            base_iou_split.append(avg_base_iou)
            random_iou_split.append(avg_random_iou)

    return lle_iou_split, lle_iou_oldmem_split, lle_iou_nomem_split, base_iou_split, random_iou_split


def main():
    print("Start.")
    args = set_parameters()

    pretrained_bert, bert_tokenizer = None, None
    if args.sim_metric == BERT_COS:
        # load bert for similarity measure
        from transformers import BertModel, BertTokenizer
        # vectorize src according to BERT pre-training models
        pretrained_bert = BertModel.from_pretrained(args.bert_dir)
        bert_tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
        if torch.cuda.is_available():
            pretrained_bert = pretrained_bert.cuda()

    lle_iou_all, base_iou_all, random_iou_all = [], [], []
    lle_oldmem_iou_all, lle_nomem_iou_all = [], []

    for t in range(len(args.dataset_config)):
        task = args.dataset_config[t]
        splits = [INF, TEST] if t > 0 else [TEST]    # test on both test and inference set (if applicable)

        if args.pgm_option == PGM_OPTION_SIMILARITY:
            for split in splits:
                (src_path, tgt_path) = get_data_files(args, task)[split]  # get the base src/tgt pairs
                src, tgt = load_src_tgt(src_path, tgt_path, clean=True)

                collect_neighbours(args, task, split, bert_tokenizer, pretrained_bert, src)
        else:
            # IoU
            lle_iou_task, base_iou_task, random_iou_task = [], [], []
            lle_oldmem_iou_task, lle_nomem_iou_task = [], []
            for split in splits:
                (src_path, tgt_path) = get_data_files(args, task)[split]  # get the base src/tgt pairs
                src, tgt = load_src_tgt(src_path, tgt_path, clean=True)

                # read the saved similarity file
                sim_file = os.path.join(args.explainer_log_dir, task, args.similarity_file[split])
                neighbour_dict = read_similarity_file(sim_file)
                lle_iou, lle_oldmem_iou, lle_nomem_iou, \
                base_iou, random_iou = measure_stability(args, task, split, neighbour_dict, src, tgt)

                lle_iou_task.extend(lle_iou)
                lle_oldmem_iou_task.extend(lle_oldmem_iou)
                lle_nomem_iou_task.extend(lle_nomem_iou)
                base_iou_task.extend(base_iou)
                random_iou_task.extend(random_iou)

            print(f'LLE IoU values on current task {task}:')
            print(lle_iou_task)
            print(f'LLE Old Memory IoU values on current task {task}:')
            print(lle_oldmem_iou_task)
            print(f'LLE No Memory IoU values on current task {task}:')
            print(lle_nomem_iou_task)
            print(f'Base IoU values on current task {task}:')
            print(base_iou_task)
            print(f'Random IoU values on current task {task}:')
            print(random_iou_task)

            lle_iou_all.append(sum(lle_iou_task) / len(lle_iou_task))
            lle_oldmem_iou_all.append(sum(lle_oldmem_iou_task) / len(lle_oldmem_iou_task))
            lle_nomem_iou_all.append(sum(lle_nomem_iou_task) / len(lle_nomem_iou_task))
            base_iou_all.append(sum(base_iou_task) / len(base_iou_task))
            random_iou_all.append(sum(random_iou_task) / len(random_iou_task))

            print('########### LLE IoU values on all tasks:')
            print(lle_iou_all)
            print('########### LLE Old Memory IoU values on all tasks:')
            print(lle_oldmem_iou_all)
            print('########### LLE No Memory IoU values on all tasks:')
            print(lle_nomem_iou_all)
            print('########### Base IoU values on all tasks:')
            print(base_iou_all)
            print('########### Random IoU values on all tasks:')
            print(random_iou_all)


if __name__ == "__main__":
    main()

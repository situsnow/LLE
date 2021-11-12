import argparse, sys

from utils.constants import DISTILBERT, AMSRV, HUGGINGFACE_SPLIT_PERCENT, HUGGINGFACE_SPLIT_ABS
from utils.utils_data import huggingface_dataset_config, get_dataset_categories, get_dataset_load_func, clean_data


def set_model_parameters():
    parser = argparse.ArgumentParser(description='Generate dict file for fairseq')
    parser.add_argument('--data-dir', type=str, help="Directory to the data")
    parser.add_argument('--cache-dir', type=str)
    # model
    parser.add_argument('--arch', type=str, default=DISTILBERT, help='the architecture of the model')
    # dataset
    parser.add_argument('--dataset', type=str, default=AMSRV)
    parser.add_argument('--split-ratio', nargs='+', default=[50, 10, 10], help='the ratio for the dataset')
    parser.add_argument('--split-unit', type=str, default=HUGGINGFACE_SPLIT_PERCENT,
                        choices=[HUGGINGFACE_SPLIT_ABS, HUGGINGFACE_SPLIT_PERCENT])

    args, _ = parser.parse_known_args(sys.argv)

    args.label_format = "<{}-{}>"
    args.label_token = "lbl"
    args.pretrained_path = "%s/%s/pretrained_%s/" % (args.data_dir, args.dataset, args.arch)
    args.pretrained_encoder_path = "%s/%s/pretrained_%s_encoder/" % (args.data_dir, args.dataset, args.arch)
    args.dict_fairseq = '%s/%s/dict.txt' % (args.data_dir, args.dataset)
    args.dataset_config = huggingface_dataset_config(args.task_order if args.dataset != AMSRV else AMSRV)
    args.categories = get_dataset_categories(args.dataset)

    return args


if __name__ == "__main__":
    args = set_model_parameters()

    vocab_dict = dict()
    for i in range(len(args.dataset_config)):
        config = args.dataset_config[i]
        train_data, _, _ = get_dataset_load_func(args.dataset)(args, config)

        x = [clean_data(i) for i in train_data['review_body']]
        y = [0 if i < 3 else 1 for i in train_data['star_rating']]   # 0: neg, 1: pos

        for c in range(len(args.categories)):
            label_cnt = sum([1 if c == j else 0 for j in y])
            try:
                vocab_dict[args.label_format.format(args.label_token, args.categories[c])] += label_cnt
            except KeyError:
                vocab_dict[args.label_format.format(args.label_token, args.categories[c])] = label_cnt

        for idx in x:
            for w in idx:
                try:
                    vocab_dict[w] += 1
                except KeyError:
                    vocab_dict[w] = 1
    sorted_vocab = sorted(vocab_dict.items(), key=lambda kv:kv[1], reverse=True)

    with open(args.dict_fairseq, 'w') as d:
        for kv in sorted_vocab:
            d.write("{} {}{}".format(kv[0], kv[1], "\n"))
import os, sys
from pathlib import Path
import torch
from transformers import BertForSequenceClassification, BertTokenizerFast, \
    DistilBertForSequenceClassification, DistilBertTokenizerFast, BertModel, DistilBertModel
from utils.constants import DISTILBERT, BERT, CHECKPOINT_PREFIX, HEURISTICS_CHECKPOINT_DICT, \
    LAST_CHECKPOINT, COPY_CHECKPOINT
from utils_functions import send_to_cuda


def find_arch_settings(arch):
    arch_dict = dict()
    arch_dict[BERT] = 'bert-base-uncased'
    arch_dict[DISTILBERT] = 'distilbert-base-uncased'
    return arch_dict[arch]


def load_tokenizer_keyencoder(args):
    if args.arch == BERT:
        huggingface_tokenizer = BertTokenizerFast
        huggingface_key_encoder = BertModel
    else:
        # use default distil BERT
        huggingface_tokenizer = DistilBertTokenizerFast
        huggingface_key_encoder = DistilBertModel
    if not os.path.exists(args.pretrained_path):
        tokenizer = huggingface_tokenizer.from_pretrained(find_arch_settings(args.arch), cache_dir=args.cache_dir)

        Path(args.pretrained_path).mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(args.pretrained_path)
    else:
        tokenizer = huggingface_tokenizer.from_pretrained(args.pretrained_path, cache_dir=args.cache_dir)

    if not os.path.exists(args.pretrained_encoder_path):
        key_encoder = huggingface_key_encoder.from_pretrained(
            find_arch_settings(args.arch), cache_dir=args.cache_dir)

        Path(args.pretrained_encoder_path).mkdir(parents=True, exist_ok=True)
        key_encoder.save_pretrained(args.pretrained_encoder_path)
    else:
        key_encoder = huggingface_key_encoder.from_pretrained(args.pretrained_encoder_path, cache_dir=args.cache_dir)

    return tokenizer, key_encoder


def load_model(args, config, lrp=False, num_labels=2):
    checkpoint_path = '%s/%s/' % (args.log_dir, config)

    # load from pretrained BERT
    if args.arch == BERT and not lrp:
        huggingface_classifier = BertForSequenceClassification
    elif args.arch == BERT and lrp:
        from models.huggingface_bert import MyBertForSequenceClassification
        huggingface_classifier = MyBertForSequenceClassification
    elif not lrp:
        # use default distil BERT
        huggingface_classifier = DistilBertForSequenceClassification
    else:
        # models
        from models.huggingface_distilbert import MyDistilBertForSequenceClassification
        huggingface_classifier = MyDistilBertForSequenceClassification
    if (not exist_checkpoint(checkpoint_path)) and config == args.dataset_config[0]:
        # 1. load pretrained model
        if (not os.path.exists(args.pretrained_path)) or \
                (not os.path.exists(os.path.join(args.pretrained_path, 'pytorch_model.bin'))):
            print('####### Load model from huggingface server #######')
            model = huggingface_classifier.from_pretrained(find_arch_settings(args.arch), cache_dir=args.cache_dir,
                                                           num_labels=num_labels)

            Path(args.pretrained_path).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.pretrained_path)
        else:
            print(f'####### Load model from local path {args.pretrained_path} #######')
            model = huggingface_classifier.from_pretrained(args.pretrained_path, cache_dir=args.cache_dir,
                                                           num_labels=num_labels)

        return model
    elif (not exist_checkpoint(checkpoint_path)) and config != args.dataset_config[0]:
        # load from previously checkpoint
        checkpoint_path = '%s/%s/' % (args.log_dir, args.dataset_config[args.dataset_config.index(config) - 1])

    # else load from current (data_config) most recent checkpoint
    checkpoint_folder = get_recent_checkpoint(checkpoint_path)
    print(f'####### Load model from checkpoint {checkpoint_folder} #######')
    model = huggingface_classifier.from_pretrained(checkpoint_folder, cache_dir=args.cache_dir,
                                                   num_labels=num_labels)
    return model


def load_explainer(args, config, distributed_training=False):
    explainer_checkpoint_path = '%s/%s/checkpoint_last.pt' % (args.explainer_log_dir, config)
    if (not exist_checkpoint(explainer_checkpoint_path)) and config != args.dataset_config[0]:
        # load from previous config
        explainer_checkpoint_path = '%s/%s' % (args.explainer_log_dir,
                                               args.dataset_config[args.dataset_config.index(config) - 1])

    trainer = load_explainer_checkpoint(args.fs_explainer_parameters, explainer_checkpoint_path, distributed_training)
    return trainer


def load_explainer_checkpoint(explainer_parameters, checkpoint_path):

    args = load_fairseq_args(explainer_parameters)

    trainer = load_fairseq_trainer(args, checkpoint_path)

    return trainer


def load_fairseq_args(explainer_parameters):

    from fairseq.fairseq import options

    load_explainer_parameters(explainer_parameters)
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)

    return args


def load_fairseq_trainer(args, checkpoint_path):
    from fairseq.fairseq import checkpoint_utils, tasks
    from fairseq.fairseq.trainer import Trainer
    task = tasks.setup_task(args)

    model = task.build_model(args)
    criterion = task.build_criterion(args)
    trainer = Trainer(args, task, model, criterion)

    if os.path.exists(checkpoint_path):
        state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint_path)
        trainer.get_model().load_state_dict(state['model'], strict=True)

    return trainer


def load_explainer_parameters(configuration_path):
    sys.argv = ""

    with open(configuration_path, 'r') as config:
        arguments = []
        for line in config:
            arguments.extend(line.split())
        sys.argv = arguments


def exist_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return False
    for f in os.listdir(checkpoint_path):
        if CHECKPOINT_PREFIX in f:
            return True
    return False


def get_recent_checkpoint(dir_path, target_checkpoint=''):
    max_checkpoint = 0
    suffix = ''
    checkpoint_prefix = CHECKPOINT_PREFIX
    for f in os.listdir(dir_path):
        if CHECKPOINT_PREFIX in f:
            # target for either last/best, or whoever comes first
            if target_checkpoint != '' and target_checkpoint in f \
                    or (target_checkpoint == '' and ('last' in f or 'best' in f)):
                return os.path.join(dir_path, f)
            # in case the target checkpoint is one of [best, last] but the file is another
            if target_checkpoint != '' and target_checkpoint in ['last', 'best']:
                continue
            end_idx = len(f)   # if the file name has no '.pt'
            if '.' in f:
                end_idx = f.index('.')
                suffix = f[end_idx:]
            if '_' in f:
                checkpoint_prefix += '_'
            chkpt = int(f[f.index(checkpoint_prefix) + len(checkpoint_prefix):end_idx])
            if chkpt > max_checkpoint:
                max_checkpoint = chkpt
    return os.path.join(dir_path, checkpoint_prefix + str(max_checkpoint) + suffix)


def prediction_by_bert(tokenizer, bbox, src):
    import torch.nn.functional as F
    with torch.no_grad():
        input_id = send_to_cuda(torch.tensor([tokenizer.encode(src)]))
        y_hat = F.softmax(bbox(input_id)[0][0], dim=0).cpu().detach().numpy()
    return y_hat


def prediction_by_bert_bulk(tokenizer, model, src=None, input_ids=None):
    import torch.nn.functional as F
    if src is not None:
        input_ids = convert_src_input_ids(src, tokenizer)
    if input_ids is not None and len(input_ids.shape) == 3:
        # the input_ids are embeddings
        outputs = F.log_softmax(model(inputs_embeds=input_ids)[0], dim=1)
    else:
        # for normal input_ids (after tokenizer)
        outputs = F.log_softmax(model(input_ids)[0], dim=1)
    del input_ids   # free memory
    return outputs


def convert_src_input_ids(src, tokenizer):
    input_id = [tokenizer.encode(' '.join(x) if type(x) == list else x) for x in src]

    max_len = len(max(input_id, key=len))
    new_input_id = []
    for x in input_id:
        while len(x) < max_len:
            x.append(tokenizer.pad_token_id)
        new_input_id.append(x)
    del input_id

    new_input_id = send_to_cuda(torch.tensor(new_input_id))
    return new_input_id


def copy_checkpoint(args, t):
    cur_chkpt_dir = os.path.join(args.explainer_log_dir, args.dataset_config[t],
                                 HEURISTICS_CHECKPOINT_DICT[args.am_heuristics])
    prev_chkpt = os.path.join(args.explainer_log_dir, args.dataset_config[t-1],
                              HEURISTICS_CHECKPOINT_DICT[args.am_heuristics], LAST_CHECKPOINT)

    Path(cur_chkpt_dir).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(prev_chkpt):
        raise RuntimeError(f'Cannot find the last checkpoint in directory {prev_chkpt}')

    os_cmd = f'cp {prev_chkpt} {cur_chkpt_dir}/{COPY_CHECKPOINT}'
    os.system(os_cmd)
    print(f'Copy checkpoint from previous task to current task with command {os_cmd}')


def remove_prev_checkpoint(args, t):
    cur_chkpt_dir = os.path.join(args.explainer_log_dir, args.dataset_config[t],
                                 HEURISTICS_CHECKPOINT_DICT[args.am_heuristics])
    prev_chkpt = os.path.join(cur_chkpt_dir, COPY_CHECKPOINT)

    if not os.path.exists(prev_chkpt):
        print(f'Checkpoint {prev_chkpt} does not exist.')
    else:
        os_cmd = f'rm {prev_chkpt}'
        os.system(os_cmd)
        print(f'Remove the duplicated checkpoint with commond {os_cmd}')


def convert_to_bert_hidden_states(bert_tokenizer, pretrained_bert, x):
    with torch.no_grad():
        x = " ".join(x).replace("<", "[").replace(">", "]")
        input_ids = torch.tensor([bert_tokenizer.encode(x)])
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        hidden_states, _ = pretrained_bert(input_ids)[-2:]
        return torch.sum(hidden_states.squeeze(0), dim=0)


def convert_to_n_gram(placeholder1, placeholder2, x):
    from nltk.util import ngrams

    all_ngrams = []

    def split_sent(x):
        if isinstance(x, list):
            return x
        else:
            return x.split(" ")
    # up till 4-gram or len(x)
    for j in range(1, min(5, len(split_sent(x))+1)):
        ngram = ngrams(split_sent(x), j)
        j_gram = []
        for each in ngram:
            j_gram.append([" ".join(each)][0] if len(each) > 1 else list(each)[0])
        all_ngrams.append(j_gram)

    return all_ngrams


def bert_cos_sim(a, b, dummy):
    from torch.nn import CosineSimilarity
    cos = CosineSimilarity(dim=0, eps=1e-6)
    return cos(a, b).item()


def n_gram_sim(a, b, logarithm=False):
    import math

    def intersect_over_union(a_ngram, b_ngram):
        # a small epsilon value to control no intersect case
        return max(len(set(a_ngram) & set(b_ngram)) / len(set(a_ngram) | set(b_ngram)), epsilon)

    sim = 0.0
    for j in range(min(len(a), len(b))):
        precision = intersect_over_union(a[j], b[j])
        # similarity result has no difference with/out logarithm
        sim += math.log2(precision) if logarithm else precision
    return sim / min(len(a), len(b))
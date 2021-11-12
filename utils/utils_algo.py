import torch, os, random

from math import floor
from utils_functions import split_sentence, join_sentence, send_to_cuda, normalize, \
    handle_task_specific_fairseq_parameters
from utils_model import prediction_by_bert_bulk, load_explainer_parameters, load_fairseq_args, \
    load_fairseq_trainer, get_recent_checkpoint
from constants import HEURISTICS_RANDOM, FAIRSEQ_OUTPUT_DICT, AMSRV, HEURISTICS_NOMEM, HEURISTICS_OLDMEM, \
    HEURISTICS_CHECKPOINT_DICT, COPY_CHECKPOINT, POS


def get_lime_weights(args, text, model, tokenizer):

    from lime.lime_text import LimeTextExplainer

    def bert_predict(inputs):
        return get_output_as_mini_batch(args, tokenizer, model, src=inputs).cpu().detach().numpy()

    text = split_sentence(text)
    num_features = len(text)
    lime_explainer = LimeTextExplainer(class_names=args.categories, random_state=args.random_state,
                                       mask_string=tokenizer.mask_token,
                                       bow=False)   # bow must set as False

    exp = lime_explainer.explain_instance(join_sentence(text), bert_predict, num_samples=args.sample_size,
                                          num_features=num_features, labels=[args.explain_label])

    dict_exp = dict(exp.as_list(label=args.explain_label))

    weights = []
    for each in text:
        try:
            w = dict_exp[each]
        except KeyError:
            w = 0.0
        weights.append(w)
    # normalize the weight
    lime_exp = list(map(lambda x: round(x, 8), normalize(weights, args.norm)))

    return lime_exp


def get_lrp_weights(args, text, input_id, logits, model, tokenizer):

    model.train()

    lrp_out_mask = send_to_cuda(torch.zeros((input_id.shape[0], len(args.categories))))
    lrp_out_mask[:, args.explain_label] = 1.0
    relevance_score = logits * lrp_out_mask
    lrp_score = model.backward_lrp(relevance_score)
    lrp_score = lrp_score.cpu().detach().data
    lrp_score = torch.abs(lrp_score).sum(dim=-1)

    lrp_score = construct_weights_after_backward(lrp_score[0], input_id, tokenizer, text)

    lrp_score = list(map(lambda x: round(x, 8), lrp_score))

    del lrp_out_mask

    return lrp_score


def get_output_as_mini_batch(args, tokenizer, model, src=None, input_ids=None):
    if src is not None and input_ids is not None:
        raise ValueError("You cannot specify both input_ids and src at the same time")
    flag = True if src is not None else False
    # split the inputs into mini-batch with size 200
    nsamples = len(src) if flag else len(input_ids)
    all_outputs = None
    i = 0

    defined_mini_batch = args.mini_batch
    while nsamples - defined_mini_batch * i > 0:
        try:
            if flag:
                mini_batch = src[defined_mini_batch * i:min(defined_mini_batch * (i + 1), nsamples)]
                output = prediction_by_bert_bulk(tokenizer, model, src=mini_batch)
                del mini_batch
            else:
                mini_batch = input_ids[defined_mini_batch * i:min(defined_mini_batch * (i + 1), nsamples)]

                output = prediction_by_bert_bulk(tokenizer, model, input_ids=mini_batch)
                del mini_batch
            if all_outputs is None:
                all_outputs = output
            else:
                all_outputs = torch.cat([all_outputs, output], dim=0)
            del output
            i += 1
        except RuntimeError:
            # reduce the mini batch size and try the same program again
            if defined_mini_batch >= 2:
                defined_mini_batch = floor(defined_mini_batch / 2)
                print(f'Adjust mini batch to {defined_mini_batch}')
                continue
            else:
                raise RuntimeError(f'Cannot process current long document (not even with mini-batch =1)')

    return all_outputs


def construct_weights_after_backward(weights, input_id, tokenizer, text):

    from transformers.tokenization_roberta import RobertaTokenizer
    from transformers.tokenization_roberta_fast import RobertaTokenizerFast
    from transformers.tokenization_distilbert import DistilBertTokenizer
    from transformers.tokenization_distilbert_fast import DistilBertTokenizerFast

    def is_roberta():
        return issubclass(type(tokenizer), (RobertaTokenizer, RobertaTokenizerFast))
    def is_distilbert():
        return issubclass(type(tokenizer), (DistilBertTokenizer, DistilBertTokenizerFast))

    def convert_ids_to_tokens():
        if hasattr(tokenizer, 'ids_to_tokens'):
            return [tokenizer.ids_to_tokens[input_id[0][idx].item()] for idx in range(len(input_id[0]))]
        else:
            tokens = [tokenizer.convert_ids_to_tokens(input_id[0][idx].item()) for idx in range(len(input_id[0]))]

            if is_distilbert():
                return tokens
            for i in range(1, len(tokens)-1):  # exclude [CLS] and [SEP]
                if is_roberta():
                    if tokens[i][0] != 'Ġ' and i != 1:
                        tokens[i] = '##' + tokens[i]
                    else:
                        tokens[i] = tokens[i].replace('Ġ', '')
                else:
                    if (tokens[i] == '▁') or (tokens[i][0] != '▁' and tokens[i-1] != '##▁'):
                        # if current token not start with _ and
                        # previous token is not '##▁' ('▁' originally, but we updated)
                        # then it belongs to previous token
                        tokens[i] = '##' + tokens[i]
                    else:
                        # otherwise, it is the start of a valid token
                        tokens[i] = tokens[i].replace('▁', '')
        return tokens

    tokens = convert_ids_to_tokens()
    if '##' not in tokens[1]:  # need to filter the case like 'ions ...' where ions considers as '-ions'
        valid_token_start = 1
    else:
        valid_token_start = 2
    new_weights = [weights[valid_token_start].item()]  # the token must be valid and does not start with ##
    for i in range(valid_token_start + 1, len(tokens)-1):  # exclude [CLS], the first valid token and [SEP]

        if '##' not in tokens[i]:
            # this is a valid token or start of a split token
            # append the weight to new_weights
            new_weights.append(weights[i].item())
        else:
            # this is part of a previous token
            new_weights[-1] += weights[i].item()
    try:
        text = split_sentence(text)
        assert len(new_weights) == len(text)
    except AssertionError:
        if len(new_weights) > len(text):
            new_weights = new_weights[:len(text)]   # exclude the padding
        else:
            # the text had been truncated, those tokens will have zero weight
            while len(new_weights) != len(text):
                new_weights.append(0.0)

    return new_weights


def copy_dict(args, task):
    dict_file = os.path.join(args.explainer_log_dir, args.dict_file)
    dir_path = os.path.join(args.explainer_log_dir, task)
    new_dict_path = os.path.join(dir_path, args.new_dict_file)

    os_cmd = f"cp {dict_file} {new_dict_path}"
    os.system(os_cmd)
    print(f'Copy the dictionary to current task directory with command: {os_cmd}')


def read_fairseq_parameters(args, t):
    # double check the heuristic and fairseq explainer
    if args.am_heuristics != HEURISTICS_RANDOM and (args.am_heuristics not in args.fs_explainer_parameters):
        raise RuntimeError('Please align the --am-heuristics and --fs-explainer-parameters')

    task = args.dataset_config[t]
    load_explainer_parameters(args.fs_explainer_parameters)

    handle_task_specific_fairseq_parameters('data', task)
    if args.fs_print_file_log:
        handle_task_specific_fairseq_parameters('--print-log-file', task + '/' + FAIRSEQ_OUTPUT_DICT[args.am_heuristics])
    handle_task_specific_fairseq_parameters('--save-dir', task + '/' +
                                            HEURISTICS_CHECKPOINT_DICT[args.am_heuristics])

    handle_task_specific_fairseq_parameters('--restore-file', COPY_CHECKPOINT, opt='assert')

    # if it is the first task, remove the memory dataset
    # or it we require no memory replay
    if (t == 0 and args.dataset == AMSRV) \
            or (args.dataset == AMSRV and HEURISTICS_NOMEM in args.fs_explainer_parameters):
        handle_task_specific_fairseq_parameters('--valid-subset', 'dev,test', opt='replace')
    elif args.dataset == AMSRV and (HEURISTICS_OLDMEM in args.fs_explainer_parameters):
        # ensure all splits are validated when OLD/EXPIRE experience relay is required
        handle_task_specific_fairseq_parameters('--valid-subset', 'dev,test,exp', opt='replace')
        # Added by Snow 27 Jul, 2021
        # use the expire memory for replay
        handle_task_specific_fairseq_parameters('--memory-split', 'exp', opt='replace')
        # Added by Snow 27 Jul, 2021
    elif args.dataset == AMSRV and (HEURISTICS_OLDMEM not in args.fs_explainer_parameters):
        # ensure all splits are validated when experience relay is required
        handle_task_specific_fairseq_parameters('--valid-subset', 'dev,test,mem', opt='replace')

    elif t == 0 or HEURISTICS_NOMEM in args.fs_explainer_parameters:
        # text classification
        handle_task_specific_fairseq_parameters('--valid-subset', 'test', opt='replace')
    elif HEURISTICS_OLDMEM in args.fs_explainer_parameters:
        handle_task_specific_fairseq_parameters('--valid-subset', 'test,exp', opt='replace')
    else:
        handle_task_specific_fairseq_parameters('--valid-subset', 'test,mem', opt='replace')


def masked_top_weight_tokens(args, src, weights, masked_token, masked_all=False, interval=0):
    def sort_weight(i):
        return weights[i]

    if masked_all and interval == 0:
        if args.cat.lower() == POS:
            weights = [0 if weights[i] > 0 else 1 for i in range(len(weights))]
        else:
            weights = [0 if weights[i] < 0 else 1 for i in range(len(weights))]
        new_src = [masked_token if weights[i - 1] == 0 else src[i] for i in range(1, len(src) - 1)]
    elif interval > 0:
        # original weights
        if args.cat.lower() == POS:
            order = sorted(range(len(weights)), key=lambda i: sort_weight(i), reverse=True)
            # multilabel
            weights = [0 if i in order[:interval] and weights[i] > 0 else 1 for i in range(len(weights))]
        else:
            order = sorted(range(len(weights)), key=lambda i: sort_weight(i))
            weights = [0 if i in order[:interval] and weights[i] < 0 else 1 for i in range(len(weights))]

        new_src = [masked_token if weights[i - 1] == 0 else src[i] for i in range(1, len(src) - 1)]
    else:
        raise RuntimeError('Please make sure either masked_all be False or interval == 0.')

    return new_src


def get_weights_from_explainer(trainer, test_example, output_dim=3):
    # import torch.nn.functional as F
    # convert the test_example according to the dictionary
    tokens = trainer.task.dictionary.encode_line(
        join_sentence(test_example), add_if_not_exist=False,
        append_eos=False, reverse_order=False,
    ).long().unsqueeze(0)

    # make the dict for the model input
    input_dict = dict()
    input_dict['src_tokens'] = send_to_cuda(tokens)
    # tokens.cuda() if torch.cuda.is_available() else tokens.clone().detach()
    input_dict['src_lengths'] = torch.tensor([len(test_example)])
    input_dict['src_text'] = None

    output = trainer.get_model()(**input_dict)[1:-1]   # exclude the front/back label tag
    if output_dim > 1:
        _, pred = torch.max(output.transpose(1, 0), 2)  # B * T
    else:
        pred = output.squeeze(2).squeeze(1)
    # transformer_output = F.log_softmax(output.float(), dim=-1)
    return pred


def masked_random_tokens(src, interval, masked_token, times=5):
    new_srcs = []
    for _ in range(times):
        masked_indices = random.sample(range(1, len(src) - 1), interval)
        new_src = [masked_token if i in masked_indices else src[i] for i in range(1, len(src) - 1)]
        new_srcs.append(new_src)

    return new_srcs


def prepare_load_fairseq_explainer(args, task, explainer_name, explainer_parameters, retain_data_path=False,
                                   target_checkpoint='last'):
    explainer_log_dir = os.path.join(args.explainer_log_dir, task, explainer_name)
    fairseq_args = load_fairseq_args(explainer_parameters)
    if not retain_data_path:
        fairseq_args.data = os.path.join(fairseq_args.data, task)
    explainer = load_fairseq_trainer(fairseq_args,
                                     get_recent_checkpoint(explainer_log_dir, target_checkpoint=target_checkpoint))
    return explainer
import argparse

def get_parser(mode='train', parser=None):
    assert mode in ['train', 'eval', 'test', 'swa', 'preprocess']
    if parser == None:
        parser = argparse.ArgumentParser(description = mode)
    get_common_args(parser)
    if mode != 'preprocess':
        get_module_args(parser)
        get_train_args(parser)
        get_beam_args(parser)
        if mode in ['eval', 'test']:
            get_eval_args(parser)
        if mode in ['swa']:
            get_swa_args(parser)
    elif mode == 'preprocess':
        get_preprocess_args(parser)
    return parser


def get_common_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('common')
    group.add_argument('--module_name', help='module description name', type=str, default='graph_rel_transformer')
    group.add_argument('--dataset_name', help='dataset save name', type=str, choices=['uspto_50k', 'uspto_MIT', 'uspto_full'], default='uspto_50k')
    group.add_argument('--split_type', help='smiles sequence split type', type=str, choices=['token', 'char'], default='token')
    group.add_argument('--save_name', help='some description of save module', type=str, default='Module 1')
    group.add_argument('--seed', help='the random seed for module running', type=int, default=17)


def get_preprocess_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('preprocess')
    group.add_argument('--raw_csv_preprocess', help='preprocess raw csv file', action='store_true', default=True)
    group.add_argument('--graph_preprocess', help='preprocess token to graph', action='store_true', default=True)
    group.add_argument('--split_preprocess', help='split eval data to smaller and accelerate evaling during training', action='store_true', default=True)
    group.add_argument('--split_len', help='the data length after spliting eval data', type=int, default=10000)
    group.add_argument('--split_name', help='the suffix of eval split data', type=str, default='split')
    group.add_argument('--need_atom', help='using smiles token to generate chemical feature',action='store_true', default=False)
    group.add_argument('--need_graph', help='using smiles token to generate molecule graph',action='store_true', default=True)
    group.add_argument('--self_loop', help='add self-loop connect to each molecule atom',action='store_true', default=False)
    group.add_argument('--vnode', help='add virtual node to molecule graph, which connect to all of the atom',
                       action='store_true', default=False)


def get_module_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('module')
    group.add_argument('--batch_size', help='the product molecule graph num for each step(same during training and evaling)', default=128)
    group.add_argument('--token_limit', help='the maximun token number of product+substrate for each step(same during training and evaling)', default=0)
    group.add_argument('--d_model', help='hidden size of module', type=int, default=256)
    group.add_argument('--d_ff', help='hidden size of feed-forward-network', type=int, default=256 * 6)
    group.add_argument('--enc_head', help='attention head of transformer encoder', type=int, default=8)
    group.add_argument('--dec_head', help='attention head of transformer decoder', type=int, default=8)
    group.add_argument('--graph_layer', help='layer size of D-MPNN', type=int, default=4)
    group.add_argument('--enc_layer', help='layer size of transformer encoder', type=int, default=6)
    group.add_argument('--dec_layer', help='layer size of transformer decoder', type=int, default=6)
    group.add_argument('--dropout', help='dropout rate of module', type=float, default=0.3)
    group.add_argument('--use_subs', help='use reaction-prediction task to help retrosynthesis, if True, the real batch_size will be a double size',
                       action='store_true', default=True)
    group.add_argument('--use_reaction_type', help='use reaction type label to help module get higher performance', action='store_true', default=False)
    group.add_argument('--decoder_cls', help='if True, in the front of decoder BOS token will have extra cls token(subs/prob or reaction_type)', default=True)
    group.add_argument('--graph_dist_block', help='the node distance block for embedding, if any distance not in this block,\
                                                   it will be included into an extra embedding',
                        type=list, default=[0, 1, 2, 3, 4, 5, 6, 7, [8, 15], [15, 2048]])
    group.add_argument('--device', help='the device for module running', type=str, default='cuda:0')

def get_train_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('training')
    group.add_argument('--train_task', help='retrosynthesis or reaction prediction task for training', choices=['prod2subs', 'subs2prod', 'bidirection'], default='bidirection')
    group.add_argument('--save_strategy', help='the checkpoint save strategy during training, if (top), it will keep checkpoint according to top1-acc;\
                                                  if (mean), it will keep checkpoint according to (top1 + mean(top3 + top5 + top10)); if (last), it will\
                                                  keep the lastest checkpoint according to epoch', type=str, choices=['top', 'mean', 'last'], default='mean')
    group.add_argument('--save_num', help='the max checkpoint num', type=int, default=10)
    group.add_argument('--swa_count', help='it will generate a average checkpoint when the save count reach the swa_count each time', type=int, default=10)
    group.add_argument('--swa_tgt', help='when reach the swa_count, it will average the top-swa_tgt checkpoint', type=int, default=5)
    group.add_argument('--const_save_epoch', help='when a checkpoint is in here, it will be keeping permanently although it already out of the save queue',
                       nargs='*', type=int, default=[])

    group.add_argument('--epochs', help='the total epochs to finish training', type=int, default=600)
    group.add_argument('--memory_clear_count', help='pytorch memory clear count in each epoch', type=int, default=0)
    group.add_argument('--eval', help='if True, module will evaling during training', action='store_true', default=True)
    group.add_argument('--save_epoch', help='when reaching these epoch, module will be saved and eval',
                       nargs='+', type=int, default=[_ for _ in range(154, 600, 5)] + [600])
    group.add_argument('--eval_epoch', help='in these epoch, module will evaling but not saving', nargs='*', type=int, default=[])
    group.add_argument('--token_eval_epoch', help='in these epoch, module will generate a file about each tokens correct rate', nargs='*', type=int, default=[])
    group.add_argument('--accum_count', help='the gradient update accum count', type=int, default=2)

    group.add_argument('--optimizer', help='the optimizer name for training', type=str, choices=['AdamW'], default='AdamW')
    group.add_argument('--lr', help='the basic learning rate scale for lr_schedule(in original, it will be a scale rate for real lr; in cosine, it will be the maximum lr)', type=float, default=1.0)
    group.add_argument('--betas', help='the betas for AdamW', nargs=2, action='append', type=int, default=[0.9, 0.999])
    group.add_argument('--eps', help='the eps for AdamW', type=float, default=1e-6)
    group.add_argument('--weight_decay', help='the weight_decay rate for AdamW', type=float, default=0.0)
    group.add_argument('--clip_norm', help='the gradient clip maximum norm, if <= 0.0, it will skip the gradient clip', type=float, default=0.0)

    group.add_argument('--ignore_min_count', help='if token occur count less than min_count, it will be ignore when calculating token weight',
                       type=int, default=100)
    group.add_argument('--label_smooth', help='the prob for negative label loss calculation', type=float, default=0.1)
    group.add_argument('--max_scale_rate', help='for the focal loss, the maximun scale rate for the less token', type=float, default=0.)
    group.add_argument('--gamma', help='the index number of difficulty weight according to the correct rate of each token in a minibatch',
                       type=float, default=2.0)
    group.add_argument('--margin', help='the rate for sentence weight calculate, if prob > margin, the result will be treated as True', type=float, default=0.85)
    group.add_argument('--sentence_scale', help='the loss ratio for sentence which is correct for all of the token', type=float, default=0.)

    group.add_argument('--warmup_step', help='the step to reach the maximum learning rate', type=int, default=8000)
    group.add_argument('--lr_schedule', help='the schedule to scale learning rate', type=str, choices=['original', 'cosine'], default='original')
    group.add_argument('--min_lr', help='when lr <= min_lr, the lr will equal to min_lr(only available in cosine lr_schedule)', type=float, default=1e-5)

def get_beam_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('beam_search')
    group.add_argument('--beam_module', help='use huggingface of OpenNMT model to running beam search, huggingface may slower 3 or more times than OpenNMT,\
                                                but OpenNMT has some bugs which cause different batch size generate different result, and also a lower accuracy.',
                       type=str, choices=['huggingface', 'OpenNMT'], default='huggingface')
    group.add_argument('--beam_size', help='the beam size for each latent variable group during prediction, because of duplicate predictions in different latent group, >=10 setting is suggested',
                       type=int, default=10)
    group.add_argument('--return_num', help='the return predictions number for each batch after beam search', type=int, default=10)
    group.add_argument('--max_len', help='the maximum length for smiles prediction', type=int, default=256)
    group.add_argument('--T', help='the tempreture for prediction log_softmax, T = 1.3 will improve performance in some cases', type=float, default=1.0)
    group.add_argument('--k_sample', help='the top-k sample setting, 0 will close top-k sample', type=int, default=0)
    group.add_argument('--p_sample', help='the top-p sample setting, 0 will close top-p sample', type=float, default=0.)
    group.add_argument('--top1_weight', help='the weight for top-1 accuracy in weighted accuracy scores', type=float, default=0.9)

def get_eval_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('evaling')
    group.add_argument('--mode', help='using eval dataset or test dataset to generate prediction', choices=['eval', 'test'], default='eval')
    group.add_argument('--eval_task', help='retrosynthesis or reaction prediction task for evaling', choices=['prod2subs', 'subs2prod'], default='prod2subs')
    group.add_argument('--ckpt_list', help='use a loop to eval the checkpoint inside this list', nargs='+', type=str, default=[])
    group.add_argument('--use_splited_data', help='use the splited data when evaling', action='store_true', default=False)
    group.add_argument('--split_data_name', help='the suffix of splited data, the default setting is (split_10000)', type=str, default='split_10000')

def get_swa_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('swa')
    group.add_argument('--average_list', help='use these checkpoint to generate an average model, it will have much better performance usually(especially in top3-10)',
                       nargs='+', type=str, default=[])
    group.add_argument('--average_name', help='the save name of this average checkpoint', type=str, default='swa')

import os
import requests
import zipfile
import argparse

from parser_loader import get_parser
from Data_preprocessing import main as preprocessing
from Module_training import training
from Module_evaling import evaling
from model.module_tools import set_seed

def running_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=str, choices=['preprocess', 'train', 'eval', 'test'])
    parser.add_argument('--dataset', type=str, choices=['50k', '50k_class', 'MIT', 'full'], default='50k')
    parser.add_argument('--download_checkpoint', action='store_true', default=False)
    args = parser.parse_args()

    get_parser(mode=args.request, parser=parser)
    return parser

def name_transform(dataset_name: str):
    if dataset_name in ['50k', '50k_class']:
        return 'uspto_50k'
    elif dataset_name == 'MIT':
        return 'uspto_MIT'
    elif dataset_name == 'full':
        return 'uspto_full'

def download_checkpoint(dataset: str):
    checkpoint_dir = os.path.dirname(os.path.realpath(__file__))
    checkpoint_dir = os.path.join(checkpoint_dir, 'model')
    checkpoint_dir = os.path.join(checkpoint_dir, 'check_point')
    checkpoint_dir = os.path.join(checkpoint_dir, dataset)

    if dataset == '50k': url = 'https://www.dropbox.com/scl/fo/vxbhzwbi38khsfqj0ed1g/h?dl=0&rlkey=b6ve5agochkgx29112kcqo5fu'
    elif dataset == '50k_class': url = 'https://www.dropbox.com/scl/fo/sum8joi7o79ktqb4xptt5/h?dl=0&rlkey=d6bncggto2i4kxrgtj71budud'
    elif dataset == 'MIT': url = 'https://www.dropbox.com/scl/fo/rjt5578efwnac6ziyx4hm/h?dl=0&rlkey=7bp7m5vpvvj0u1x1wlonqdypo'
    elif dataset == 'full': url = 'https://www.dropbox.com/scl/fo/4x88221gk5oju5jgblbp6/h?dl=0&rlkey=7yb1d0vgxhhyz0r8zz93lsejf'
    
    checkpoint_path = os.path.join(checkpoint_dir, 'data.zip')
    if not os.path.exists(checkpoint_path):
        r = requests.get(url, stream=True)
        with open(checkpoint_path, 'wb') as f:
            for i in r.iter_content(chunk_size=128):
                f.write(i)
        with zipfile.ZipFile(checkpoint_path) as f:
            f.extractall(path=checkpoint_dir)


if __name__ == '__main__':
    parser = running_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    if args.request == 'preprocess':
        args.raw_csv_preprocess = True
        args.graph_preprocess = True
        args.split_preprocess = False if args.dataset in ['50k', '50k_class'] else True
        args.split_len = 10000
        args.dataset_name = name_transform(args.dataset)
        preprocessing(args)

    elif args.request == 'train':
        args.save_name = args.dataset
        args.dataset_name = name_transform(args.dataset)
        args.accum_count = 2
        args.use_subs = True
        args.use_reaction_type = True if args.dataset in ['50k_class'] else False
        args.decoder_cls = True
        args.save_strategy = 'mean'
        args.epochs = 700 if args.dataset in ['50k', '50k_class'] else 100
        args.save_epoch = [_ for _ in range(149, args.epochs, 5)] + [args.epochs] if args.dataset in ['50k', '50k_class']\
            else [_ for _ in range(19, args.epochs, 1)] + [args.epochs]
        args.batch_size = 128 if args.dataset in ['50k', '50k_class'] else 64
        args.token_limit = 12000 if args.dataset in ['full'] else 0
        args.memory_clear_count = 1 if args.dataset in ['50k', '50k_class'] else 4
        args.lr = 1 if args.dataset in ['50k', '50k_class'] else 1.25
        args.dropout = 0.3 if args.dataset in ['50k', '50k_class'] else 0.1
        args.train_task = 'bidirection'
        args.eval_task = 'subs2prod' if args.dataset in ['MIT'] else 'prod2subs'
        args.use_splited_data = False if args.dataset in ['50k', '50k_class'] else True
        args.split_data_name = 'split_10000'
        training(args)
    
    elif args.request in ['eval', 'test']:
        if args.download_checkpoint:
            download_checkpoint(args.dataset)
            args.ckpt_list = [args.dataset]

        args.save_name = args.dataset
        args.dataset_name = name_transform(args.dataset)
        args.mode = args.request
        args.use_subs = True
        args.use_reaction_type = True if args.dataset in ['50k_class'] else False
        args.beam_module = 'huggingface'
        args.batch_size = 128 if args.dataset in ['50k', '50k_class'] else 64
        args.token_limit = 12000 if args.dataset in ['full'] else 0
        args.beam_size = 20
        if args.dataset in ['50k', '50k_class']:
            args.T = 1.6
        elif args.dataset in ['MIT']:
            args.T = 1.1
        elif args.dataset in ['full']:
            args.T = 0.7
        args.eval_task = 'subs2prod' if args.dataset in ['MIT'] else 'prod2subs'
        args.max_len = 512
        evaling(args)

import os

from parser_loader import get_parser
from model.torch_data_loader import FullDataset
from model.module_tools import set_seed, Model_Save

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


def averaging(args):
    eval_dataset = FullDataset(
        dataset_name=args.dataset_name,
        split_type=args.split_type,
        batch_size=args.batch_size,
        mode='eval',
        dist_block=args.graph_dist_block
    )
    ckpt_dir = os.path.join(eval_dataset.check_point_path, args.save_name)
    module_saver = Model_Save(
        ckpt_dir=ckpt_dir,
        device=args.device,
        save_strategy=args.save_strategy,
        save_num=args.save_num,
        swa_count=args.swa_count,
        swa_tgt=args.swa_tgt,
        const_save_epoch=args.const_save_epoch,
        top1_weight=args.top1_weight
    )
    module_saver.swa(args.average_list, args.average_name)


if __name__ == '__main__':
    parser = get_parser(mode='swa')
    args = parser.parse_args()
    args.save_name = ''
    args.average_list = []
    args.average_name = 'swa'

    set_seed(args.seed)
    averaging(args)

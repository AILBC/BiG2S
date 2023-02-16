import os
import logging

from parser_loader import get_parser
from model.preprocess.uspto_50k import uspto50k
from model.preprocess.uspto_MIT import uspto_MIT
from model.preprocess.uspto_full import uspto_full
from model.preprocess.seq_tokenize import reaction_sequence
from model.preprocess.token_featurelize import Data_Preprocess
from model.preprocess.eval_split import Data_Split


def main(args):
    raw_csv_reader = None
    csv_preprocess = None
    if args.raw_csv_preprocess:
        if args.dataset_name == 'uspto_50k': raw_csv_reader = uspto50k()
        elif args.dataset_name == 'uspto_MIT': raw_csv_reader = uspto_MIT()
        elif args.dataset_name == 'uspto_full': raw_csv_reader = uspto_full()

        logging.basicConfig(filename=os.path.join(raw_csv_reader.dir, 'preprocess.log'),
                            format='%(asctime)s %(message)s', level=logging.INFO)
        for k, v in args.__dict__.items():
            logging.info('args -> {0}: {1}'.format(k, v))
        
        raw_csv_reader.csv_process()
        logging.info('csv/txt preprocess finish.')

        csv_preprocess = reaction_sequence(
            dataset_name=args.dataset_name,
            split_type=args.split_type
        )
        csv_preprocess.get_vocab_dict()
        csv_preprocess.smi_encode()
        logging.info('maximum reaction length: {0}'.format(csv_preprocess.max_len))
        logging.info('average reaction length: {0}'.format(csv_preprocess.avg_len))
        logging.info('total reaction data count: {0}'.format(csv_preprocess.reaction_count))
        logging.info('\n')

    if args.graph_preprocess:
        graph_preprocess = Data_Preprocess(
            dataset_name=args.dataset_name,
            split_type=args.split_type,
            need_atom=args.need_atom,
            need_graph=args.need_graph,
            self_loop=args.self_loop,
            vnode=args.vnode
        )
    
    if args.split_preprocess:
        eval_split = Data_Split(
            dataset_name=args.dataset_name,
            split_type=args.split_type,
            seed=args.seed
        )
        eval_split.split(
            split_len=args.split_len,
            split_name=args.split_name
        )


if __name__ == '__main__':
    parser = get_parser(mode='preprocess')
    args = parser.parse_args()
    # args.raw_csv_preprocess = True
    # args.graph_preprocess = True
    # args.split_preprocess = False #please enable it when training on USPTO-MIT and USPTO-full for more efficient evaluation  
    # args.split_len = 10000
    # args.dataset_name = 'uspto_50k'

    main(args)
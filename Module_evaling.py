import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.std import trange

from model.graph_rel_transformer import GraphTransformer
from parser_loader import get_parser
from model.preprocess.chem_preprocess import NODE_FDIM, BOND_FDIM
from model.torch_data_loader import FullDataset
from model.module_tools import Model_Save, set_seed, eval_plot, beam_result_process

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


def evaling(args):
    if args.use_subs and args.use_reaction_type:
        dec_cls = 2
    elif args.use_subs or args.use_reaction_type:
        dec_cls = 1
    else:
        dec_cls = 0
    eval_dataset = FullDataset(
        dataset_name=args.dataset_name,
        split_type=args.split_type,
        batch_size=args.batch_size,
        token_limit=args.token_limit,
        mode=args.mode,
        dist_block=args.graph_dist_block,
        task=args.eval_task,
        use_split=args.use_splited_data,
        split_data_name=args.split_data_name
    )
    ckpt_dir = os.path.join(eval_dataset.check_point_path, args.save_name)
    token_idx = eval_dataset.token_idx
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
    module = GraphTransformer(
        f_vocab=len(token_idx),
        f_atom=NODE_FDIM + 10 if args.use_reaction_type else NODE_FDIM,
        f_bond=BOND_FDIM,
        token_idx=token_idx,
        token_freq=eval_dataset.token_freq,
        token_count=eval_dataset.token_count,
        cls_len=dec_cls if args.decoder_cls else 0,
        args=args
    )

    for ckpt_name in args.ckpt_list:
        _, module.model, _, _ = module_saver.load(ckpt_name, module.model)
        beam_size = args.beam_size
        seq_acc_count = np.zeros((args.return_num))
        seq_invalid_count = np.zeros((args.return_num))
        reaction_acc_count = np.zeros((10))
        smi_predictions = []

        with torch.no_grad():
            eval_dataset.get_batch()
            data_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=1,
                shuffle=True,
                collate_fn=lambda _batch: _batch[0],
                pin_memory=True
            )
            teval = trange(eval_dataset.batch_step)
            for step, batch in zip(teval, data_loader):
                batch = batch.to(args.device)

                predict_result, predict_scores = module.model_predict(
                    data=batch,
                    args=args
                )
                beam_acc, beam_invalid, beam_smi = beam_result_process(
                    tgt_seq=batch.tgt_seq,
                    tgt_len=batch.tgt_seq_len,
                    token_idx=token_idx,
                    beam_result=predict_result,
                    beam_scores=predict_scores
                )
                seq_acc_count = seq_acc_count + beam_acc
                seq_invalid_count = seq_invalid_count + beam_invalid
                smi_predictions.extend(beam_smi)

                teval.set_description('evaling......')

            seq_acc_count = np.cumsum(seq_acc_count)
            seq_invalid_count = np.cumsum(seq_invalid_count)
            seq_acc_count = seq_acc_count / eval_dataset.data_len
            seq_invalid_count = seq_invalid_count / np.array([i * eval_dataset.data_len for i in range(1, args.return_num + 1, 1)])

            eval_plot(
                topk_seq_acc=seq_acc_count.tolist(),
                topk_seq_invalid=seq_invalid_count.tolist(),
                beam_size=beam_size,
                data_name=args.dataset_name,
                ckpt_dir=ckpt_dir,
                ckpt_name=ckpt_name,
                args=args,
                is_train=False
            )
            

if __name__ == '__main__':
    parser = get_parser(mode = 'eval')
    args = parser.parse_args()
    # args.dataset_name = 'uspto_50k'
    # args.save_name = '50k'
    # args.mode = 'test'
    # args.use_subs = True
    # args.use_reaction_type = False
    # args.decoder_cls = True
    # args.ckpt_list = ['50k']
    # args.beam_module = 'huggingface'
    # args.batch_size = 128
    # args.token_limit = 0
    # args.beam_size = 20
    # args.T = 1.6
    # args.eval_task = 'prod2subs'
    # args.max_len = 512

    set_seed(args.seed)
    evaling(args)

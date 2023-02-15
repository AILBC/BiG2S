import os
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

from model.preprocess.smiles_tools import canonicalize_smiles, smi2token
from model.preprocess.chem_preprocess import NODE_FDIM, BOND_FDIM
from model.preprocess.token_featurelize import smi2graph
from model.torch_data_loader import BatchData, graph_batch
from model.graph_rel_transformer import GraphTransformer
from parser_loader import get_parser
from model.module_tools import set_seed

class one_step_analysis():
    def __init__(
        self,
        args,
        vocab_dir: str,
        module_dir: str
    ):  
        vocab_list = []
        with open(vocab_dir, 'r') as f:
            for token in f:
                vocab_list.append(token.split('\t')[0])
        vocab_dict = {token:idx for idx, token in enumerate(vocab_list)}
        index_token = {j:i for i, j in vocab_dict.items()}
        self.vocab_dict = vocab_dict
        self.index_token = index_token

        if args.use_reaction_type:
            dec_cls = 2
        else:
            dec_cls = 1
        predict_module = GraphTransformer(
            f_vocab=len(vocab_dict),
            f_atom=NODE_FDIM + 10 if args.use_reaction_type else NODE_FDIM,
            f_bond=BOND_FDIM,
            token_idx=self.vocab_dict,
            token_freq=None,
            token_count=None,
            cls_len=dec_cls,
            args=args
        )
        module_ckpt = torch.load(module_dir, map_location=args.device)
        predict_module.model.load_state_dict(module_ckpt['module_param'])
        self.module = predict_module
    
    def _beam_result_process(
        self,
        beam_result: torch.Tensor,
        beam_scores: torch.Tensor
    ):
        eos_ids, pad_ids = self.vocab_dict['<EOS>'], self.vocab_dict['<PAD>']
        beam_result = beam_result.detach().cpu().numpy()
        beam_scores = beam_scores.detach().cpu().numpy()
        all_smi = []
        for batch_id, batch_res in enumerate(beam_result):
            beam_smi = []
            for beam_id, beam_res in enumerate(batch_res):
                res = beam_res[((beam_res != eos_ids) & (beam_res != pad_ids))]
                res_smi = [self.index_token[idx] for idx in res]
                res_smi = ''.join(res_smi)
                res_smi = canonicalize_smiles(res_smi, False, False)
                if res_smi == '': res_smi = 'CC'
                beam_smi.append(res_smi)
            beam_smi = '\t'.join(beam_smi)
            all_smi.append(beam_smi)
        return all_smi, beam_scores
    
    def _preprocess(
        self,
        tgt_dir: str,
        args
    ):
        bos_id, eos_id, pad_id, unk_id = self.vocab_dict.get('<BOS>'), self.vocab_dict.get('<EOS>'),\
            self.vocab_dict.get('<PAD>'), self.vocab_dict.get('<UNK>')
        with open(tgt_dir, 'r') as f:
            tgt_list, tgt_save = [], []
            tgt_task_list = []
            seq_len = []
            for data in f:
                data = data.strip('\n')
                smi, task = data.split('\t')
                tgt_save.append(smi)
                tgt_task_list.append(int(task))
                smi = canonicalize_smiles(smi=smi)
                smi = smi2token(smi)
                smi = smi.split(' ')
                smi.append('<EOS>')
                seq_len.append(len(smi))
                smi_encode = [self.vocab_dict.get(x, unk_id) for x in smi]
                tgt_list.append(smi_encode)

        tgt_task_list = np.array(tgt_task_list, dtype=np.int64)
        if (tgt_task_list == 0).all(): predict_task = 'prod2subs'
        elif (tgt_task_list == 1).all(): predict_task = 'subs2prod'
        else:
            assert sum(tgt_task_list == 0) == sum(tgt_task_list == 1)
            assert (tgt_task_list[:len(tgt_task_list) // 2] == 0).all()
            predict_task = 'bidirection'
        seq_max_len = max(seq_len)
        tgt_list = [smi + (seq_max_len - len(smi)) * [pad_id] for smi in tgt_list] 
        tgt_list = np.stack(tgt_list, axis=0)

        tgt_graph = [smi2graph(smi, self.index_token, is_subs=tgt_task_list[i]) for i, smi in enumerate(tgt_list)]
        graph_len = [graph['graph_atom'].shape[0] for graph in tgt_graph]
        graph_max_len = max(graph_len)
        tgt_graph = graph_batch(
            graphs=tgt_graph,
            graph_len=graph_len,
            max_len=graph_max_len,
            dist_block=args.graph_dist_block
        )
        reaction_type = np.array([0 for _ in range(tgt_list.shape[0])], dtype=np.int64)

        tgt_seq, seq_len = torch.tensor(tgt_list, dtype=torch.long),\
            torch.tensor(seq_len, dtype=torch.long)
        tgt_graph['node_feat'], tgt_graph['node_connect'], tgt_graph['bond_feat'],\
        tgt_graph['node_dist'], tgt_graph['node_deg'], graph_len, tgt_graph['bond_neighbour'] =\
            torch.tensor(tgt_graph['node_feat'], dtype=torch.long), torch.tensor(tgt_graph['node_connect'], dtype=torch.long),\
            torch.tensor(tgt_graph['bond_feat']), torch.tensor(tgt_graph['node_dist'], dtype=torch.long),\
            torch.tensor(tgt_graph['node_deg'], dtype=torch.long), torch.tensor(graph_len, dtype=torch.long),\
            torch.tensor(tgt_graph['bond_neighbour'], dtype=torch.long)
        reaction_type = torch.tensor(reaction_type, dtype=torch.long)
        bi_label = torch.tensor(tgt_task_list, dtype=torch.long)
        batch_data = BatchData(
            tgt_seq=tgt_seq,        # no use
            tgt_seq_len=seq_len,    # no use
            src_graph=tgt_graph,
            src_graph_len=graph_len,
            reaction_type=reaction_type,
            bi_label=bi_label,
            task=predict_task
        )
        return batch_data, tgt_save, tgt_task_list

    def predict(
        self,
        tgt_dir: str,
        tgt_name: str,
        args,
        plot_num = 3
    ):
        batch_data, tgt_save, tgt_task_list = self._preprocess(
            tgt_dir=tgt_dir,
            args=args
        )
        with torch.no_grad():
            batch_data = batch_data.to(args.device)
            predict_result, predict_scores = self.module.model_predict(
                data=batch_data,
                args=args
            )
            predict_result, predict_scores = self._beam_result_process(
                beam_result=predict_result,
                beam_scores=predict_scores
            )
        raw_dir = os.path.dirname(os.path.realpath(__file__))
        save_dir = os.path.join(raw_dir, tgt_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        with open(os.path.join(save_dir, 'predict_result.txt'), 'w') as f:
            for i in range(len(tgt_task_list)):
                f.writelines('{id}.task:{task}\t source:{smi}\n'.format(id=i+1, task=tgt_task_list[i], smi=tgt_save[i]))
                result = predict_result[i]
                result = result.split('\t')
                scores = predict_scores[i]
                for j in range(len(result)):
                    f.writelines('{id}.:\t {score:.4}\t {smi}\n'.format(id=j+1, score=scores[j], smi=result[j]))
                f.writelines('\n')
        for i in range(len(tgt_task_list)):
            result = predict_result[i]
            result = result.split('\t')
            result = [Chem.MolFromSmiles(smi) for smi in result]
            svg_draw = rdMolDraw2D.MolDraw2DSVG(300, 300)
            svg_draw.ClearDrawing()
            rdMolDraw2D.PrepareAndDrawMolecule(svg_draw, Chem.MolFromSmiles(tgt_save[i]))
            svg_draw.FinishDrawing()
            with open(os.path.join(save_dir, '{batch}-source.svg'.format(batch=i+1)), 'w') as f:
                f.write(svg_draw.GetDrawingText())
            for j in range(plot_num):
                svg_draw = rdMolDraw2D.MolDraw2DSVG(300, 300)
                svg_draw.ClearDrawing()
                rdMolDraw2D.PrepareAndDrawMolecule(svg_draw, result[j])
                svg_draw.FinishDrawing()
                with open(os.path.join(save_dir, '{batch}-{beam}.svg'.format(batch=i+1, beam=j+1)), 'w') as f:
                    f.write(svg_draw.GetDrawingText())


if __name__ == '__main__':
    parser = get_parser(mode = 'eval')
    args = parser.parse_args()
    tgt_dir = 'one_step_tgt.txt'
    tgt_name = 'one_step_tgt'
    vocab_dir = 'vocab_full.txt'
    module_dir = 'full.ckpt'

    args.use_reaction_type = False
    args.beam_module = 'huggingface'
    args.beam_size = 20
    args.max_len = 512
    args.T = 0.8

    set_seed(args.seed)
    one_step_predict = one_step_analysis(
        args=args,
        vocab_dir=vocab_dir,
        module_dir=module_dir
    )
    one_step_predict.predict(
        tgt_dir=tgt_dir,
        tgt_name=tgt_name,
        args=args,
        plot_num=10
    )


import os
import copy
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict,List
from torch.utils.data import DataLoader, Dataset

from .preprocess.dataset_basic import DataLoad
from .preprocess.chem_preprocess import MAX_DIST, MAX_DEG
from .preprocess.token_featurelize import token_to_upper


def graph_to_seq(node_feat: torch.Tensor, node_num: torch.Tensor, max_len=None, remove_vnode=False):
    if max_len == None:
        max_len = node_num.max()
    pad_node = []
    split_node = torch.split(node_feat, node_num.tolist(), dim=0)
    for length, node in zip(node_num, split_node):
        operate = nn.ZeroPad2d((0, 0, 0, max_len - length))
        if remove_vnode:
            pad_node.append(operate(node)[1:])
        else:
            pad_node.append(operate(node))

    pad_node = torch.stack(pad_node, dim=0)
    return pad_node


def seq_to_graph(node_feat: torch.Tensor, node_num: torch.Tensor, max_len=None):
    '''unpad graph'''
    if max_len == None:
        max_len = node_num.max()
    unpad_node = []
    for length, node in zip(node_num, node_feat):
        unpad_node.append(node[:length])
    
    unpad_node = torch.cat(unpad_node, dim=0)
    return unpad_node


def bond_to_seq(bond: torch.Tensor, connect: torch.Tensor, node_num: torch.Tensor, bond_batch_idx: List[int], add_vnode=False):
    max_len = node_num.max().item()
    node_num = node_num.tolist()
    node_num_shift = torch.cumsum(torch.tensor([0] + node_num[:-1],
                                  dtype=torch.long, device=bond.device), dim=0)[bond_batch_idx]
    connect = connect - node_num_shift.unsqueeze(0)
    if add_vnode:
        bond_pad = torch.zeros(len(node_num), max_len + 1, max_len + 1, 
                               bond.size(-1), dtype=bond.dtype, device=bond.device)
        connect = connect + 1
    else:
        bond_pad = torch.zeros(len(node_num), max_len, max_len, bond.size(-1),
                               dtype=bond.dtype, device=bond.device)
    bond_idx = [bond_batch_idx, connect[0].tolist(), connect[1].tolist()]
    bond_pad[bond_idx] = bond
    return bond_pad


def sequence_pad(seq, seq_len, max_len, pad):
    seq_copy = copy.deepcopy(seq)
    for i, length in enumerate(seq_len):
        seq_copy[i] += [pad] * (max_len - length)
    return np.stack(seq_copy, axis=0)


def batch_pad(atom, connect, bond, bond_neighbour):
    atom_num = atom.shape[0]
    bond_num = bond.shape[0]
    atom = np.pad(atom, ((1, 0), (0, 0)), 'constant')

    connect_pad_src = np.zeros((atom_num), dtype=np.int64)
    connect_pad_tgt = np.arange(1, atom_num + 1, 1)
    connect_pad = np.stack([connect_pad_src, connect_pad_tgt], axis=0)
    connect = np.concatenate([connect_pad, connect + 1], axis=1)

    bond = np.pad(bond, ((atom_num, 0), (0, 0)), 'constant')

    bond_neighbour_pad1 = np.arange(atom_num, atom_num + bond_num, 1)
    bond_neighbour_pad2 = connect[0][atom_num:]
    bond_neighbour_pad3 = np.searchsorted(connect_pad_tgt, bond_neighbour_pad2, 'left')
    bond_neighbour_pad = np.stack([bond_neighbour_pad1, bond_neighbour_pad2, bond_neighbour_pad3], axis=0)
    bond_neighbour[0] += atom_num
    bond_neighbour[1] += 1
    bond_neighbour[2] += atom_num
    bond_neighbour = np.concatenate([bond_neighbour_pad, bond_neighbour], axis=1)

    return atom, connect, bond, bond_neighbour


def graph_batch(graphs, graph_len, max_len, cls_len=0, dist_block=None):
    assert isinstance(dist_block, list)
    batch_node_feat = []
    batch_node_connect = []
    batch_bond_feat = []
    batch_node_dist = []
    batch_node_deg = []
    batch_node_path = []
    batch_bond_neighbour = []
    batch_idx = 0
    bond_batch_list = 0
    bias = 0
    bond_bias = 0

    for g, length in zip(graphs, graph_len):
        graph = copy.deepcopy(g)
        node_feat, node_connect, bond_feat, node_dist, node_deg, node_path, bond_neighbour = graph['graph_atom'],\
            graph['graph_connect'], graph['graph_bond'], graph['graph_dist'], graph['graph_deg'], graph['graph_path'], graph['graph_bond_neighbour']
        bond_num = len(bond_feat)
        if cls_len > 0:
            for i in range(cls_len):
                node_dist = np.pad(node_dist, ((1, 0), (1, 0)),
                               'constant', constant_values=1)
                node_dist[0, 0] = 0  
        node_dist = np.pad(node_dist, ((0, max_len - length), (0, max_len - length)),
                           'constant', constant_values=-1)
                   
        batch_node_dist = np.concatenate([batch_node_dist, np.expand_dims(node_dist, 0)], axis=0) if\
            isinstance(batch_node_dist, np.ndarray) else np.expand_dims(node_dist, 0)
        if len(node_feat) <= 1:
            node_feat = np.expand_dims(node_feat, 0)
        batch_node_feat = np.concatenate([batch_node_feat, node_feat], axis=0) if\
            isinstance(batch_node_feat, np.ndarray) else node_feat
        if len(node_deg) <= 1:
            node_deg = np.expand_dims(node_deg, 0)
        batch_node_deg = np.concatenate([batch_node_deg, node_deg], axis=0) if\
            isinstance(batch_node_deg, np.ndarray) else node_deg

        if isinstance(node_connect, np.ndarray):
            # node_connect --> [src_node_idx, tgt_node_idx, reverse_edge_idx]
            node_connect[0] += bias
            node_connect[1] += bias
            node_connect[2] += bond_bias
            batch_node_connect = np.concatenate([batch_node_connect, node_connect], axis=1) if\
                isinstance(batch_node_connect, np.ndarray) else node_connect
        if isinstance(bond_feat, np.ndarray):
            batch_bond_feat = np.concatenate([batch_bond_feat, bond_feat], axis=0) if\
                isinstance(batch_bond_feat, np.ndarray) else bond_feat
        if isinstance(bond_neighbour, np.ndarray):
            # bond_neighbour --> [i->j edge idx, k->i edge idx]
            bond_neighbour[0] += bond_bias
            bond_neighbour[1] += bond_bias
            batch_bond_neighbour = np.concatenate([batch_bond_neighbour, bond_neighbour], axis=1) if\
                isinstance(batch_bond_neighbour, np.ndarray) else bond_neighbour
        if isinstance(bond_batch_list, int):
            bond_batch_list = [batch_idx] * bond_num
        else:
            bond_batch_list.extend([batch_idx] * bond_num)

        batch_idx += 1
        bias += length
        bond_bias += bond_num

    batch_node_connect = batch_node_connect.astype(np.int64)
    for id, blk in enumerate(dist_block):
        if isinstance(blk, list):
            batch_node_dist[(batch_node_dist >= blk[0]) & (batch_node_dist < blk[1])] = id
        else:
            batch_node_dist[batch_node_dist == blk] = id
    min_dist = dist_block[0][0] if isinstance(dist_block[0], list) else dist_block[0]
    max_dist = dist_block[-1][-1] if isinstance(dist_block[-1], list) else dist_block[-1]
    batch_node_dist[(batch_node_dist < min_dist) | (batch_node_dist > max_dist)] = len(dist_block)

    return {'node_feat': batch_node_feat, 'node_connect': batch_node_connect, 'bond_feat': batch_bond_feat, 'node_dist': batch_node_dist,
            'node_deg': batch_node_deg, 'node_path': batch_node_path, 'bond_batch_idx': bond_batch_list, 'bond_neighbour': batch_bond_neighbour}


class BatchData():
    def __init__(
        self,
        tgt_seq: torch.Tensor,
        tgt_seq_len: torch.Tensor,
        src_graph: Dict,
        src_graph_len: torch.Tensor,
        reaction_type: torch.Tensor,
        bi_label=None,
        pair_label=None,
        task='prod2subs'
    ):
        self.tgt_seq = tgt_seq
        self.tgt_seq_len = tgt_seq_len
        self.src_node, self.src_link, self.src_bond, self.src_dist, self.src_deg, self.src_path, self.src_bond_idx, self.src_bond_neighbour =\
            [_ for _ in src_graph.values()]
        self.src_graph_len = src_graph_len
        self.reaction_type = reaction_type
        self.bi_label = bi_label
        self.pair_label = pair_label
        self.task = task

    def to(self, device):
        self.tgt_seq = self.tgt_seq.to(device)
        self.tgt_seq_len = self.tgt_seq_len.to(device)
        self.src_node, self.src_link, self.src_bond, self.src_dist, self.src_deg, self.src_graph_len, self.src_bond_neighbour =\
            self.src_node.to(device), self.src_link.to(device), self.src_bond.to(device),\
            self.src_dist.to(device), self.src_deg.to(device), self.src_graph_len.to(device),\
            self.src_bond_neighbour.to(device)
        self.reaction_type = self.reaction_type.to(device)
        self.bi_label = self.bi_label.to(device) if\
            self.bi_label is not None else self.bi_label
        self.pair_label = self.pair_label.to(device) if\
            self.pair_label is not None else self.pair_label
        return self

    def pin_memory(self):
        self.tgt_seq = self.tgt_seq.pin_memory()
        self.tgt_seq_len = self.tgt_seq_len.pin_memory()
        self.src_node, self.src_link, self.src_bond, self.src_dist, self.src_deg, self.src_graph_len, self.src_bond_neighbour =\
            self.src_node.pin_memory(), self.src_link.pin_memory(), self.src_bond.pin_memory(),\
            self.src_dist.pin_memory(), self.src_deg.pin_memory(), self.src_graph_len.pin_memory(),\
            self.src_bond_neighbour.pin_memory()
        self.reaction_type = self.reaction_type.pin_memory()
        self.bi_label = self.bi_label.pin_memory() if\
            self.bi_label is not None else self.bi_label
        self.pair_label = self.pair_label.pin_memory() if\
            self.pair_label is not None else self.pair_label
        return self


class DataMemory(DataLoad):
    def __init__(
        self,
        dataset_name,
        split_type,
        load_type='data',
        module_mode='train',
        dist_block=None,
        use_split=False,
        split_data_name='',
        cls_len=0
    ):
        super(DataMemory, self).__init__(
            dataset_name=dataset_name,
            split_type=split_type,
            load_type=load_type,
            mode=module_mode,
            use_split=use_split,
            split_data_name=split_data_name
        )
        self.mode = module_mode
        self.length_list = self.length_data
        self.token_idx = self.vocab_data['token_idx']
        self.token_freq = self.vocab_data['token_freq']
        self.token_count = self.vocab_data['token_count']
        self.data_len = self.data[self.mode]['subs']['token'].shape[0]
        self.dist_block = dist_block
        self.cls_len = cls_len

    def to_tensor(self, data, dtype=None):
        if isinstance(data, np.ndarray) or isinstance(data, list):
            data = torch.tensor(data) if\
                dtype is None else torch.tensor(data, dtype=dtype)
        return data

    def get_batch_data(self, batch_idx: np.ndarray, task='prod2subs') -> BatchData:
        if not isinstance(batch_idx, np.ndarray):
            batch_idx = np.array(batch_idx)
        subs_data = self.data[self.mode]['subs']
        prod_data = self.data[self.mode]['prod']
        reaction_type = self.data[self.mode]['reaction_type']

        if task == 'prod2subs':
            tgt_seq = copy.deepcopy(subs_data['token'][batch_idx])
            bi_label = np.array([0] * batch_idx.shape[0])
        elif task == 'subs2prod':
            tgt_seq = copy.deepcopy(prod_data['token'][batch_idx])
            bi_label = np.array([1] * batch_idx.shape[0])
        tgt_seq_len = [len(seq) for seq in tgt_seq]
        seq_maxlen = max(tgt_seq_len)
        tgt_seq = sequence_pad(
            seq=tgt_seq,
            seq_len=tgt_seq_len,
            max_len=seq_maxlen,
            pad=self.token_idx['<PAD>']
        )

        if task == 'prod2subs':
            src_graph = copy.deepcopy(prod_data['graph'][batch_idx])
        elif task == 'subs2prod':
            src_graph = copy.deepcopy(subs_data['graph'][batch_idx])
        src_graph_len = [graph['graph_atom'].shape[0] for graph in src_graph]
        graph_maxlen = max(src_graph_len)
        src_graph = graph_batch(
            graphs=src_graph,
            graph_len=src_graph_len,
            max_len=graph_maxlen,
            cls_len=self.cls_len,
            dist_block=self.dist_block
        )

        tgt_seq, tgt_seq_len = self.to_tensor(tgt_seq, torch.long),\
            self.to_tensor(tgt_seq_len, torch.long)

        src_graph['node_feat'], src_graph['node_connect'], src_graph['bond_feat'],\
        src_graph['node_dist'], src_graph['node_deg'], src_graph_len, src_graph['bond_neighbour'] =\
            self.to_tensor(src_graph['node_feat'], torch.long), self.to_tensor(src_graph['node_connect'], torch.long),\
            self.to_tensor(src_graph['bond_feat']), self.to_tensor(src_graph['node_dist'], torch.long),\
            self.to_tensor(src_graph['node_deg'], torch.long), self.to_tensor(src_graph_len, torch.long),\
            self.to_tensor(src_graph['bond_neighbour'], torch.long)
        reaction_type = self.to_tensor(reaction_type[batch_idx], torch.long)
        bi_label = self.to_tensor(bi_label, torch.long)

        return_data = BatchData(
            tgt_seq=tgt_seq,
            tgt_seq_len=tgt_seq_len,
            src_graph=src_graph,
            src_graph_len=src_graph_len,
            reaction_type=reaction_type,
            bi_label=bi_label,
            pair_label=None,
            task=task
        )

        return return_data

    def get_batch_data_with_subs(self, batch_idx: np.ndarray) -> BatchData:
        if not isinstance(batch_idx, np.ndarray):
            batch_idx = np.array(batch_idx)
        subs_data = self.data[self.mode]['subs']
        prod_data = self.data[self.mode]['prod']
        reaction_type = self.data[self.mode]['reaction_type']

        subs_seq, prod_seq = copy.deepcopy(subs_data['token'][batch_idx]),\
            copy.deepcopy(prod_data['token'][batch_idx])
        subs_seq_len, prod_seq_len = [len(seq) for seq in subs_seq],\
            [len(seq) for seq in prod_seq]
        seq_maxlen = max(max(subs_seq_len), max(prod_seq_len))
        subs_seq = sequence_pad(
            seq=subs_seq,
            seq_len=subs_seq_len,
            max_len=seq_maxlen,
            pad=self.token_idx['<PAD>']
        )
        prod_seq = sequence_pad(
            seq=prod_seq,
            seq_len=prod_seq_len,
            max_len=seq_maxlen,
            pad=self.token_idx['<PAD>']
        )
        tgt_seq = np.concatenate([subs_seq, prod_seq], axis=0)
        tgt_seq_len = subs_seq_len + prod_seq_len
        bi_label = np.array([0] * len(subs_seq_len) + [1] * len(prod_seq_len))
        pair_label = np.array([_ for _ in range(len(subs_seq_len))] * 2)
        # shuffle
        # shuffle_idx = np.random.permutation(len(tgt_seq_len))
        # bi_label = bi_label[shuffle_idx]
        # pair_label = pair_label[shuffle_idx]
        # tgt_seq, tgt_seq_len = tgt_seq[shuffle_idx], np.array(tgt_seq_len)[shuffle_idx]

        assert subs_data['graph'].any() is not None
        prod_graph = copy.deepcopy(prod_data['graph'][batch_idx])
        prod_graph_len = [graph['graph_atom'].shape[0] for graph in prod_graph]
        subs_graph = copy.deepcopy(subs_data['graph'][batch_idx])
        subs_graph_len = [graph['graph_atom'].shape[0] for graph in subs_graph]
        graph_maxlen = max(max(prod_graph_len), max(subs_graph_len))
        src_graph = np.concatenate([prod_graph, subs_graph], axis=0)
        src_graph_len = prod_graph_len + subs_graph_len
        #shuffle
        # src_graph, src_graph_len = src_graph[shuffle_idx], np.array(src_graph_len)[shuffle_idx]
        src_graph = graph_batch(
            graphs=src_graph,
            graph_len=src_graph_len,
            max_len=graph_maxlen,
            cls_len=self.cls_len,
            dist_block=self.dist_block
        )

        tgt_seq, tgt_seq_len = self.to_tensor(tgt_seq, torch.long),\
            self.to_tensor(tgt_seq_len, torch.long)
        src_graph['node_feat'], src_graph['node_connect'], src_graph['bond_feat'],\
        src_graph['node_dist'], src_graph['node_deg'], src_graph_len, src_graph['bond_neighbour'] =\
            self.to_tensor(src_graph['node_feat'], torch.long), self.to_tensor(src_graph['node_connect'], torch.long),\
            self.to_tensor(src_graph['bond_feat']), self.to_tensor(src_graph['node_dist'], torch.long),\
            self.to_tensor(src_graph['node_deg'], torch.long), self.to_tensor(src_graph_len, torch.long),\
            self.to_tensor(src_graph['bond_neighbour'], torch.long)
        reaction_type = self.to_tensor(reaction_type[batch_idx], torch.long)
        reaction_type = reaction_type.repeat(2)
        bi_label = self.to_tensor(bi_label, torch.long)
        pair_label = self.to_tensor(pair_label, torch.long)
        #shuffle
        # reaction_type = reaction_type[shuffle_idx]

        return_data = BatchData(
            tgt_seq=tgt_seq,
            tgt_seq_len=tgt_seq_len,
            src_graph=src_graph,
            src_graph_len=src_graph_len,
            reaction_type=reaction_type,
            bi_label=bi_label,
            pair_label=pair_label,
            task='bidirection'
        )

        return return_data


class FullDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        split_type='token',
        batch_size=64,
        token_limit=0,
        mode='train',
        dist_block=None,
        task='prod2subs',
        use_split=False,
        split_data_name='',
        cls_len=0
    ):
        self.datamemory = DataMemory(
            dataset_name=dataset_name,
            split_type=split_type,
            module_mode=mode,
            dist_block=dist_block,
            use_split=use_split,
            split_data_name=split_data_name,
            cls_len=cls_len
        )
        self.mode = mode
        self.task = task
        self.token_idx = self.datamemory.token_idx
        self.idx_token = {v: k for k, v in self.token_idx.items()}
        self.idx_token_up = token_to_upper(self.idx_token)
        self.token_freq = self.datamemory.token_freq
        self.token_count = self.datamemory.token_count
        self.data_len = self.datamemory.data_len
        self.data_len_list = self.datamemory.length_list
        self.all_token_count = sum(self.data_len_list)
        self.batch_order = []
        self.batch_size = batch_size
        self.token_limit = token_limit
        self.batch_len = self._get_batch_len()

    def _get_batch_len(self):
        batch_len = [self.batch_size] * (self.data_len // self.batch_size)
        batch_len = batch_len + [self.data_len % self.batch_size] if\
            self.data_len % self.batch_size != 0 else batch_len
        batch_len = np.array(batch_len)
        return batch_len

    def get_batch(self):
        self.batch_order = []
        if self.token_limit == 0: # running batch according to fixed batch size
            batch_idx = np.random.permutation(self.data_len)
            self.batch_order = np.split(batch_idx, np.cumsum(self.batch_len)[:-1])
            # in training, use previous data to pad the last step to ensure each step has the same batch size
            if len(self.batch_order[-1]) < self.batch_size and self.mode == 'train':
                gap = self.batch_size - len(self.batch_order[-1])
                self.batch_order[-1] = np.concatenate([self.batch_order[-1], self.batch_order[0][:gap]], axis=0)
        
        elif self.token_limit > 0: # running batch according to flexible maximun token count
            total_idx = np.random.permutation(self.data_len)
            data_len_list = self.data_len_list[total_idx]
            idx_src = 0
            while 1:
                max_len = data_len_list[idx_src]
                batch_idx = [total_idx[idx_src]]
                assert max_len <= self.token_limit
                for idx_tgt in range(idx_src + 1, self.data_len + 1, 1):
                    if idx_tgt >= self.data_len: break
                    if len(batch_idx) > self.batch_size and self.batch_size > 0: break
                    new_max_len = max(max_len, data_len_list[idx_tgt])
                    if new_max_len * (len(batch_idx) + 1) > self.token_limit: break
                    batch_idx.append(total_idx[idx_tgt])
                    max_len = new_max_len
                self.batch_order.append(batch_idx)
                idx_src = idx_tgt + 1
                if idx_src >= self.data_len: break

    def __getitem__(self, index: int):
        idx = self.batch_order[index]
        return self.datamemory.get_batch_data(idx, task=self.task) if self.task != 'bidirection'\
            else self.datamemory.get_batch_data_with_subs(idx)

    def __len__(self):
        return self.batch_step

    @property
    def batch_step(self):
        return math.ceil(self.data_len / self.batch_size) if self.token_limit == 0\
            else len(self.batch_order)

    @property
    def check_point_path(self) -> str:
        file_path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(file_path, 'check_point')


if __name__ == '__main__':
    bd = FullDataset('uspto_50k', 'token', 32)
    bd.get_batch()
    bd_loader = DataLoader(
        dataset=bd,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda _batch: _batch[0],
        pin_memory=True
    )
    for _, batch in enumerate(bd_loader):
        batch
        
        

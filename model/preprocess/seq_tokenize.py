import os
import collections
import numpy as np
from tqdm import tqdm
from typing import Dict,List
from matplotlib import pyplot as plt

from .dataset_basic import DataLoad


def space_split(lines: List[str]) -> List[str]:
    """
    split each symbol according to space.
    """
    return lines.split()


def token_count(lines: List[str]) -> Dict[str, int]:
    """
    collect each token and its appear count.
    """
    return collections.Counter(lines)


def sequence_encode(seq: List[str], code: Dict[str, int], eos_label: str) -> List[int]:
    """
    generate sequence encoding according to dict.
    """
    seq.append(eos_label)
    return [code.get(x) for x in seq]


class reaction_sequence(DataLoad):

    def __init__(
        self, 
        dataset_name, 
        split_type='token', 
        load_type='csv'
    ):
        super(reaction_sequence, self).__init__(
            dataset_name=dataset_name, 
            split_type=split_type, 
            load_type=load_type
        )
        self.symbol_dict = {}
        self.avg_len = 0
        self.reaction_count = 0
        self.max_len = 0

    def get_vocab_dict(self, extra_token=[]):
        """
        generate vocab list according to preprocess csv data.
        symbol_dict -> Dict[symbol, symbol_code]
        """
        symbol_list = []
        static_symbol_list = ['<BOS>', '<EOS>', '<PAD>',
                              '<UNK>', '<SEP>', '<PROD>', '<SUBS>']
        static_symbol_freq = [0 for i in range(len(static_symbol_list + extra_token))]
        for data in self.data.values():
            for subs_smi, prod_smi in zip(data['subs'], data['prod']):
                subs_smi, prod_smi = subs_smi.split(), prod_smi.split()
                self.max_len = (len(subs_smi) + len(prod_smi)) if\
                     (len(subs_smi) + len(prod_smi)) > self.max_len else self.max_len
                self.avg_len += (len(subs_smi) + len(prod_smi))
                self.reaction_count += 1
                symbol_list.extend(subs_smi)
                symbol_list.extend(prod_smi)
                static_symbol_freq[1] += 2

        symbol_count = sorted(token_count(symbol_list).items(),
                              key=lambda x: x[1], reverse=True)
        symbol, symbol_freq = [x[0] for x in symbol_count],\
            [x[1] for x in symbol_count]
        symbol = static_symbol_list + extra_token + symbol
        symbol_freq = static_symbol_freq + symbol_freq
        self.vocab_freq_plot(symbol, symbol_freq)

        self.symbol_dict = {j: i for i, j in enumerate(symbol)}
        self.total_symbol_count = sum(symbol_freq)
        self.symbol_freq = [i / self.total_symbol_count for i in symbol_freq]
        self.symbol_count = symbol_freq
        self.max_len += 1
        self.avg_len += self.reaction_count
        self.avg_len = self.avg_len / self.reaction_count

    def smi_encode(self, save=True, max_len=None):
        if max_len == None:
            max_len = self.max_len
        for data, data_split in zip(self.data.values(), self.data.keys()):
            subs_enc = []
            prod_enc = []
            reaction_type = []
            reaction_len = []

            for subs_smi, prod_smi, react_type in tqdm(zip(data['subs'], data['prod'], data['reaction_type']),
                                                       desc=f'{self.dataset_name} is now encoding...', total=len(data['subs'])):
                subs_smi, prod_smi = space_split(subs_smi),\
                    space_split(prod_smi)
                subs_enc.append(sequence_encode(subs_smi, self.vocab_dict, '<EOS>')[:])
                prod_enc.append(sequence_encode(prod_smi, self.vocab_dict, '<EOS>')[:])
                reaction_len.append(len(subs_enc[-1]) + len(prod_enc[-1]))
                reaction_type.append(react_type)

            if save == True:
                self.encode_save(np.array(subs_enc, dtype=object), np.array(prod_enc, dtype=object),
                                 reaction_type, data_split)
                self.length_save(np.array(reaction_len), data_split)

        if save == True:
            self.dict_save()

    def encode_save(self, subs, prod, reaction_type, data_split):
        np.savez(os.path.join(self.dir, f'{data_split}_{self.split_type}_encoding.npz'),
                 subs=subs, prod=prod, reaction_type=reaction_type)

    def dict_save(self):
        np.savez(os.path.join(self.dir, f'{self.split_type}_vocab_dict.npz'), vocab_dict=self.vocab_dict,
                 seq_len=self.max_len, vocab_freq=self.symbol_freq, vocab_count=self.symbol_count)
    
    def length_save(self, len, data_split):
        np.savez(os.path.join(self.dir, f'{data_split}_{self.split_type}_length.npz'),
                 len=len)

    @property
    def vocab_dict(self) -> Dict[str, int]:
        """
        return dataset's vocab dict, which likes {symbol : symbol_code}.
        """
        return self.symbol_dict

    def vocab_freq_plot(self, symbol: List, symbol_freq: List, need_plot=False):
        """
        plot the vocab occurrence frequence.
        """
        vocab, freq = symbol, symbol_freq
        if need_plot:
            plt.bar(vocab, freq)
            plt.title('smiles symbol occurrence frequence')
            plt.xlabel('smiles symbol')
            plt.ylabel('occurrence frequence')
            plt.savefig(os.path.join(self.dir, 'token_count.png'))
            plt.close()

        with open(os.path.join(self.dir, f'{self.split_type}_token_count.txt'), 'w') as f:
            for i, j in zip(vocab, freq):
                f.writelines('{0}\t{1}\n'.format(i, j))
        

if __name__ == '__main__':
    uspto50k_sequence = reaction_sequence('uspto_50k', split_type = 'token')
    uspto50k_sequence.get_vocab_dict()
    uspto50k_sequence.smi_encode()

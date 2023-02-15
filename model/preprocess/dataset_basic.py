import os
import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod

DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw_csv')


class Dataset(metaclass=ABCMeta):
    """
    baisc class for dataset.
    """

    def __init__(self):
        super(Dataset, self).__init__()
        self._create_directory()

    def _create_directory(self):
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)

    @property
    @abstractmethod
    def key(self) -> str:
        """
        return the dataset name, such as 'uspto_50k_without_reaction_type'.
        """
        raise NotImplementedError('Abstract method')

    @property
    @abstractmethod
    def dir(self) -> str:
        """
        return the save path of preprocess dataset's folder. 
        """
        raise NotImplementedError('Abstract method')


class DataLoad(Dataset):
    """
    basic class for preprocess data load.
    """

    def __init__(self, dataset_name, split_type='token', load_type=None, mode=None, use_split=False, split_data_name=''):
        self.dataset_name = dataset_name
        self.split_type = split_type
        self.use_split = use_split
        self.split_data_name = split_data_name
        super(DataLoad, self).__init__()
        self.data = {'train': None, 'eval': None, 'test': None}
        if mode == None:
            self.data = {'train': self._get_data('train', load_type),
                         'eval': self._get_data('eval', load_type),
                         'test': self._get_data('test', load_type)}
        elif mode in ['train', 'eval', 'test']:
            self.data[mode] = self._get_data(mode, load_type)
        self.vocab_data = None
        if load_type in ['npz', 'data']:
            self.vocab_data = self._get_vocab_data()
        if load_type == 'data':
            self.length_data = self._get_smi_length(mode)

    def _get_data(self, data_split_type, load_type=None):
        """
        load preprocess data from specific datatype. load_type = ['csv', 'npz', 'data']
        """
        csv_path = os.path.join(self.dir, f'{data_split_type}.csv')
        npz_path = os.path.join(self.dir, f'{data_split_type}_{self.split_type}_encoding.npz')
        if self.use_split:
            data_path = os.path.join(self.dir, f'{data_split_type}_{self.split_type}_preprocess_{self.split_data_name}.data.npz')
        else:
            data_path = os.path.join(self.dir, f'{data_split_type}_{self.split_type}_preprocess.data.npz')
        if load_type != None:
            if load_type == 'csv':
                load_data = pd.read_csv(csv_path)
                subs_str, prod_str = 'substrates_' + self.split_type, 'product_' + self.split_type
                subs_data, prod_data, reaction_type = load_data[subs_str],\
                    load_data[prod_str], load_data['reaction_type']

            elif load_type == 'npz':
                np_data = np.load(npz_path, allow_pickle=True)
                subs_data, prod_data, reaction_type = np_data['subs'],\
                    np_data['prod'], np_data['reaction_type']

            elif load_type == 'data':
                np_data = np.load(data_path, allow_pickle=True)
                subs_data, prod_data, reaction_type = np_data['subs'].tolist(),\
                    np_data['prod'].tolist(), np_data['reaction_type']

            return {'subs': subs_data, 'prod': prod_data, 'reaction_type': reaction_type}

        else:
            raise ValueError('load_type must in [\'csv\', \'npz\', \'data\']')

    def _get_vocab_data(self):
        """
        load vocab data for preprocess.
        """
        vocab_path = os.path.join(self.dir, f'{self.split_type}_vocab_dict.npz')
        vocab_dict = np.load(vocab_path, allow_pickle=True)
        token_index, seq_max_len, token_freq, token_count = vocab_dict['vocab_dict'],\
            vocab_dict['seq_len'], vocab_dict['vocab_freq'], vocab_dict['vocab_count']
        return {'token_idx': token_index.tolist(), 'max_len': seq_max_len,
                'token_freq': token_freq.tolist(), 'token_count': token_count.tolist()}
    
    def _get_smi_length(self, mode='train'):
        """
        load smi length for variable batch size.
        """
        if self.use_split:
            length_path = os.path.join(self.dir, f'{mode}_{self.split_type}_length_{self.split_data_name}.npz')
        else:
            length_path = os.path.join(self.dir, f'{mode}_{self.split_type}_length.npz')
        length_data = np.load(length_path, allow_pickle=True)
        smi_length = length_data['len']
        return smi_length

    @property
    def key(self) -> str:
        return self.dataset_name

    @property
    def dir(self) -> str:
        return os.path.join(DATA_DIR, self.key)

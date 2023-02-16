import os
import pandas as pd
from tqdm import tqdm

from .dataset_basic import Dataset, DATA_DIR, RAW_DATA_DIR
from .smiles_tools import canonicalize_smiles, token_preprocess, char_preprocess
from .download_data import download

class uspto_MIT(Dataset):
    def __init__(self):
        super(uspto_MIT, self).__init__()

    @property
    def key(self) -> str:
        return 'uspto_MIT'

    @property
    def dir(self) -> str:
        return os.path.join(DATA_DIR, self.key)

    def csv_process(self):
        train = {
            'reaction_type': [],
            'product_token': [],
            'substrates_token': [],
            'product_char': [],
            'substrates_char': []
        }
        eval = {
            'reaction_type': [],
            'product_token': [],
            'substrates_token': [],
            'product_char': [],
            'substrates_char': []
        }
        test = {
            'reaction_type': [],
            'product_token': [],
            'substrates_token': [],
            'product_char': [],
            'substrates_char': []
        }

        for data_split_type, filename, data_store in (('train', 'train.txt', train), ('eval', 'valid.txt', eval), ('test', 'test.txt', test)):
            data_path = os.path.join(RAW_DATA_DIR, f'uspto_MIT/{filename}')
            if not os.path.exists(data_path):
                if not os.path.exists(data_path):
                    download(
                        url='https://www.dropbox.com/scl/fo/sd7rzl9tc93akjubd7kg1/h?dl=0&rlkey=0c36a844yjsgclfxe7agt8ejq',
                        save_dir=os.path.join(RAW_DATA_DIR, 'uspto_MIT'),
                        file_name='data.zip'
                    )
            raw_data = open(data_path, 'r')

            for reaction_smile in tqdm(raw_data, desc=f'split {filename} SMILES...'):
                reaction_smile = reaction_smile.strip('\n')
                reaction_smile, others = tuple(reaction_smile.split(' '))
                subs, prod = tuple(reaction_smile.split('>>'))
                subs, prod = canonicalize_smiles(subs),\
                    canonicalize_smiles(prod)
                subs_token, prod_token = token_preprocess(subs, prod)
                subs_char, prod_char = char_preprocess(subs, prod)
                data_store['reaction_type'].append(0)
                data_store['product_token'].append(prod_token)
                data_store['substrates_token'].append(subs_token)
                data_store['product_char'].append(prod_char)
                data_store['substrates_char'].append(subs_char)

            pd.DataFrame(data_store).to_csv(os.path.join(self.dir, f'{data_split_type}.csv'), index=False)


if __name__ == '__main__':
    data = uspto_MIT()
    data.csv_process()

import os
import pandas as pd
from tqdm import tqdm

from .dataset_basic import Dataset, DATA_DIR, RAW_DATA_DIR

class uspto_full(Dataset):
    def __init__(self):
        super(uspto_full, self).__init__()

    @property
    def key(self) -> str:
        return 'uspto_full'

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

        for data_split_type, data_store in (('train', train), ('eval', eval), ('test', test)):
            prod_path, subs_path = os.path.join(RAW_DATA_DIR, f'uspto_full/src-{data_split_type}.txt'),\
                os.path.join(RAW_DATA_DIR, f'uspto_full/tgt-{data_split_type}.txt')
            if not (os.path.exists(prod_path) and os.path.exists(subs_path)):
                raise FileNotFoundError(
                    f'raw data type {data_split_type} file is not exist. Please download from Graph2SMILES source code and extract to required location.'
                )
            prod_data, subs_data = open(prod_path, 'r'), open(subs_path, 'r')

            for (prod_smi, subs_smi) in tqdm(zip(prod_data, subs_data), desc=f'split uspto_full {data_split_type} SMILES...'):
                if len(prod_smi) <= 1 or len(subs_smi) <= 1: continue
                data_store['reaction_type'].append(0)
                data_store['product_token'].append(prod_smi)
                data_store['substrates_token'].append(subs_smi)
                data_store['product_char'].append(' ')
                data_store['substrates_char'].append(' ')
            
            pd.DataFrame(data_store).to_csv(os.path.join(self.dir, f'{data_split_type}.csv'), index=False)


if __name__ == '__main__':
    data = uspto_full()
    data.csv_process()
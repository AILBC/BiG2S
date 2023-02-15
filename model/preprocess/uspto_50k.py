import os
import pandas as pd
from tqdm import tqdm

from .dataset_basic import Dataset, DATA_DIR, RAW_DATA_DIR
from .smiles_tools import canonicalize_smiles, token_preprocess, char_preprocess


class uspto50k(Dataset):
    def __init__(self):
        super(uspto50k, self).__init__()

    @property
    def key(self) -> str:
        return 'uspto_50k'

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

        for data_split_type, filename, data_store in (('train', 'raw_train.csv', train), ('eval', 'raw_val.csv', eval), ('test', 'raw_test.csv', test)):
            data_path = os.path.join(RAW_DATA_DIR, f'uspto_50k/{filename}')
            if not os.path.exists(data_path):
                raise FileNotFoundError(
                    f'raw data file is not exist in {data_path}. Please download from GLN source code and extract to required location.'
                )
            raw_data = pd.read_csv(data_path)

            for reaction_smile, reaction_type in tqdm(zip(raw_data['reactants>reagents>production'], raw_data['class']),
                                                      desc=f'split {filename} SMILES...', total=len(raw_data)):
                subs, prod = tuple(reaction_smile.split('>>'))
                subs, prod = canonicalize_smiles(subs),\
                    canonicalize_smiles(prod)
                subs_token, prod_token = token_preprocess(subs, prod)
                subs_char, prod_char = char_preprocess(subs, prod)
                data_store['reaction_type'].append(reaction_type)
                data_store['product_token'].append(prod_token)
                data_store['substrates_token'].append(subs_token)
                data_store['product_char'].append(prod_char)
                data_store['substrates_char'].append(subs_char)

            pd.DataFrame(data_store).to_csv(os.path.join(self.dir, f'{data_split_type}.csv'), index=False)


if __name__ == '__main__':
    data = uspto50k()
    data.csv_process()

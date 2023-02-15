# BiG2S: A Dual Task Graph-to-Sequence Model for the End-to-End Template-Free Reaction Predicting

Here is the code for *"BiG2S: A Dual Task Graph-to-Sequence Model for the End-to-End Template-Free Reaction Predicting"*

***The manuscript is under submission now, please cite this page if you find our work is helpful***

## 1. Overview
The details of implementation environment and args settings are provided as `.txt` files, you may need to paste them to the scripts and replace the original args settings which is in:

```
if __name__ == '__main__':
    parser = get_parser(mode = 'train')
    args = parser.parse_args()
    args settings...
```

The code, datasets, checkpoints and settings need to be downloaded and pasted manually because we have not provided any `.sh` or `.yaml` file for setup.

The overview of the directory is shown as:
```
BiG2S
│  condalist.txt            #the details for conda environment
│  Data_preprocessing.py    #script for data preprocessing
│  Module_averaging.py      #script for checkpoint averaging
│  Module_evaling.py        #script for model evaling
│  Module_training.py       #script for model training
│  One_Step_Analysis.py     #dual-task prediction and visualization for the given reactions
│  one_step_tgt.txt         #paste reactions and dual-task label here for prediction
│  parser_loader.py         #manage all of the args settings
│  vocab_50k.txt
│  vocab_full.txt
│  vocab_MIT.txt            #vocabulary for One_Step_Analysis.py
│  
├─args_settings             #details about the args settings for each datasets
│      uspto_50k.txt
│      uspto_full.txt
│      uspto_MIT.txt
│      
└─model
    │  graph_encoder.py         #code for graph encoder
    │  graph_rel_transformer.py #code for structure combination
    │  module_tools.py          #code for each infrastructure (e.g. attention block)
    │  torch_data_loader.py     #batch data loader
    │  transformer_decoder.py   #code for transformer encoder
    │  transformer_encoder.py   #code for transformer decoder
    │  
    ├─check_point               #save path for each model checkpoints
    │  ├─50k
    │  ├─50k_class
    │  ├─full
    │  └─MIT
    ├─inference                 #code for each beam search algorithm
    │    hugging_face_beamsearch.py
    │    hugging_face_infer.py
    │    infer_base.py
    │    onmt_beamsearch.py
    │    onmt_infer.py
    │          
    └─preprocess                #code for data preprocessing
        │  chem_preprocess.py
        │  dataset_basic.py
        │  eval_split.py
        │  seq_tokenize.py
        │  smiles_tools.py
        │  token_featurelize.py
        │  uspto_50k.py
        │  uspto_full.py
        │  uspto_MIT.py
        │  
        └─data                  #save path for each datasets and preprocessed datasets
            └─raw_csv
                ├─uspto_50k
                ├─uspto_full
                └─uspto_MIT
```

## 2. Environment setup
Code was run and evaluated for:

    - python 3.10.8
    - pytorch 1.13.0
    - pytorch-scatter 2.1.0
    - PyG 2.2.0
    - rdkit 2022.09.1

Models were trained on RTX A5000 with 24GB memory for larger batch size(e.g. 128\*2), which also available for less GPU memory with an appropriate batch size setting and larger gradient accumulation steps(e.g. 32\*2 and accumulate 4 steps for 6GB).

Note that an earlier version of rdkit(e.g. 2019) may result in different SMILES canonicalization results.

*For more details about the conda environment, please check the `condalist.txt`.*

## 3. Data preprocessing
We mainly use **USPTO-50k**, **USPTO-full** and **USPTO-MIT** datasets for training and evaluation.

**USPTO-50k** is obtained it from: https://github.com/Hanjun-Dai/GLN, which is in `.csv` format.
**USPTO-full** is directly obtained from: https://github.com/coleygroup/Graph2SMILES, they provided a preprocessed `.txt` data that had already been splited into tokens with a regular regex.
**USPTO-MIT** is obtained from: https://github.com/pschwllr/MolecularTransformer, we used the **mixed** version which was also in `.txt` format.

When these datasets are downloaded from the link, please unpacked them to the following folder:
```
model/preprocess/data/raw_csv/uspto_50k  for USPTO-50k
model/preprocess/data/raw_csv/uspto_full  for USPTO-full
model/preprocess/data/raw_csv/uspto_MIT  for USPTO-MIT
```

Then run the data preprocessing program by
```
python Data_preprocessing.py
```
Please replace the `args.dataset_name` at the end of the code with the corresponding dataset name in advance(`uspto_50k` is the default, you can choose `uspto_full` or `uspto_MIT`. Please ensure the `args.split_preprocess` is set to `True` when choosing `uspto_full` or `uspto_MIT` for more efficient evaluation during training).

For example, the args settings for **USPTO-50k** in `Data_preprocessing.py` are like these:

```
args.raw_csv_preprocess = True  #raw dataset file --> SMILES tokenize 
args.graph_preprocess = True    #SMILES tokenize --> Molecular graph feature
args.split_preprocess = False   #True for **USPTO-MIT** and **USPTO-full**, more efficient evaluation during training
args.split_len = 10000          #the size of splited evaluation dataset 
args.dataset_name = 'uspto_50k' #dataset name
```

***Additionally, we provide the download link for the three datasets mentioned above, you can download them manually at: https://www.dropbox.com/sh/aa41sxlte7wngiv/AADWe3XEg2C1wBbfz3lktyjFa?dl=0***

## 4. Training and validation during training
We provide the hyperparameter settings in `args_settings/`, for example, the args settings for **USPTO-50k** training are as follows:

```
args.save_name = '50k'          #the folder name for checkpoints and log
args.dataset_name = 'uspto_50k' #dataset name
args.accum_count = 2            #gradient accumulation steps
args.use_subs = True            #use dual-task training
args.use_reaction_type = False  #only **USPTO-50k** can set it to True for additional reaction type information
args.decoder_cls = True         #dual-task labels and reaction type labels for decoder
args.save_strategy = 'mean'     #calculate average accuracy for each checkpoint and sort them by this metric
args.epochs = 700               #maximum epochs for training
args.save_epoch = [_ for _ in range(154, args.epochs, 5)] + [args.epochs]   #model will be saved in these epochs
args.batch_size = 128           #chemical reaction number for each batch (the real size will *2 if use dual-task training)
args.token_limit = 0            #maximum token counts for each batch
args.memory_clear_count = 1     #clean the memory in each epoch
args.d_model = 256              #hidden size&embedding size
args.d_ff = 1536                #filter size
args.enc_head = 8               #encoder attention heads
args.dec_head = 8               #decoder attention heads
args.graph_layer = 4            #D-MPNN layer
args.enc_layer = 6              #encoder layer
args.dec_layer = 6              #decoder layer
args.lr = 1                     #scale rate for learning rate schedule
args.dropout = 0.3              #dropout rate
args.train_task = 'bidirection' 
args.eval_task = 'prod2subs'    #'prod2subs' will input products; 'subs2prod' will input reactants; 'bidirection' will input both of them
```

Choose and copy the settings corresponding to the dataset and paste them into the end of the training script `Module_training.py` to replace the default args setting. When using the appropriate parameters, the model will be trained as we described in the paper by running the following training script:

```
python Module_training.py
```

You can find the training log and checkpoints at `model/check_point`.

The automatic weight averaging is available in our model, which can automatically generate weight average models based on weighted accuracy on the validation set. You can acquire multiple weight average models from the different training steps of the model after it has been trained, as well as the original models and the accuracy of each model. Use this information to quickly select the best model checkpoint for subsequent evaluation.

## 5. Evaluation for one-step reaction predicting
We also provide the evaluation hyperparameter settings in `args_settings/`, for example, the args settings for **USPTO-50k** are as follows:

```
args.dataset_name = 'uspto_50k' #dataset name
args.save_name = '50k'          #the folder name for checkpoints and log
args.mode = 'test'              #'eval' for validation set, and 'test' for test set
args.use_subs = True            #use dual-task training
args.use_reaction_type = False  #only **USPTO-50k** can set it to True for additional reaction type information
args.decoder_cls = True         #dual-task labels and reaction type labels for decoder
args.ckpt_list = ['50k']        #the model checkpoints inside will be evaluated one by one
args.beam_module = 'huggingface'    #you can choose 'huggingface' and 'OpenNMT' for different beam search algorithm
args.batch_size = 128           #chemical reaction number for each batch
args.token_limit = 0            #maximum token counts for each batch
args.beam_size = 20             #beam width for beam search
args.T = 1.6                    #temperature for Softmax, each task&dataset has an appropriate temperature setting
args.eval_task = 'prod2subs'    #'prod2subs' will input products; 'subs2prod' will input reactants; 'bidirection' will input both of them
args.max_len = 512              #the maximum token length for prediction
```

***For the model checkpoints for each datasets, you can download them manually at: https://www.dropbox.com/sh/uekt2uacoawz81g/AABIc4w-pQAWxiYk4geSgMBpa?dl=0, each of the checkpoint need to be pasted into `model/check_point`.***

Choose and copy the settings corresponding to the dataset and paste them into the end of the evaling script `Module_evaling.py` to replace the default args setting. Please replace `args.ckpt_list` with the checkpoint file name you have selected. For example, if you choose the checkpoint `model/check_point/50k/ckpt1.ckpt`, the args settings are as follows:

```
args.save_name = '50k'
args.ckpt_list = ['ckpt1']
```

We construct a new beam search method based on **huggingface**(https://github.com/huggingface/transformers), which can provide better top-n prediction quality but takes more search time, and we further improve its search speed and enable it to support group beam search and results reordering with additional latent variables. You can also use **OpenNMT**(https://github.com/OpenNMT/OpenNMT) for beam search by adding `args.beam_module = 'OpenNMT'` in `Module_evaling.py`. For more details about the beam search algorithms, please check the code in `model/inference`

Additional, the hyperparameter `T` can significantly affect the prediction results in different datasets and tasks, please select an appropriate `T` for evaluation, you can manually adjust it in `args.T`.

When choosing the appropriate setting, you can run the evaluation on different test sets by

```
python Module_evaling.py
```
which will report the accuracy and invalid rate from top-1 to top-10 by default.


We also provide a dual-task evaluation and visualization script in `One_Step_Analysis.py`, the main settings are as follows:

tgt_dir: The *path* of the `txt` file containing the molecular SMILES to be predicted and its dual-task label <br>
tgt_name: The *path* of the results output folder <br>
vocab_dir: The *path* of the tokenizer vocabulary <br>
module_dir: The *path* of the model checkpoint <br>

You can paste the `full.ckpt` checkpoint for **USPTO-full** into the root directory for dual-task evaluation, which is the same as the settings we use in the manuscript, the vocabulary is already available at `vocab_full.txt`.

Please enter the molecules and tasks to be predicted in `tgt_name` by following the format below:

```
Source SMILES   Dual-task label(0--retrosynthesis 1--forward synthesis)
CC(=O)c1ccc2c(ccn2C(=O)OC(C)(C)C)c1 0
CC(=O)c1ccc2[nH]ccc2c1.CC(C)(C)OC(=O)OC(=O)OC(C)(C)C    0
CC(=O)c1ccc2c(ccn2C(=O)OC(C)(C)C)c1 1
CC(=O)c1ccc2[nH]ccc2c1.CC(C)(C)OC(=O)OC(=O)OC(C)(C)C    1
```

Then run the script by
```
python One_Step_Analysis.py
```

Now you can find the corresponding top-10 prediction results of these four tasks and the visualization results from RDKit in `tgt_name/`.
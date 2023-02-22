# BiG2S: A Dual Task Graph-to-Sequence Model for the End-to-End Template-Free Reaction Prediction

Here is the code for *"BiG2S: A Dual Task Graph-to-Sequence Model for the End-to-End Template-Free Reaction Prediction"*

***The manuscript is under submission now, please cite this page if you find our work is helpful***

## 1. Overview
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
    - PyG 2.2.0
    - rdkit 2022.09.1
    - matplotlib 3.6.2

You can create an isolation conda environment for BiG2S by running:

```
conda create -c conda-forge -n BiG2S rdkit
conda activate BiG2S
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install pyg -c pyg
conda install matplotlib
```

Models were trained on RTX A5000 with 24GB memory for larger batch size(e.g. 128\*2), which also available for less GPU memory with an appropriate batch size setting and larger gradient accumulation steps(e.g. 32\*2 and accumulate 4 steps for 6GB).

Note that an earlier version of rdkit(e.g. 2019) may result in different SMILES canonicalization results.

*For more details about the conda environment, please check the `condalist.txt`.*

## 3. Data preprocessing
We mainly use **USPTO-50k**, **USPTO-full** and **USPTO-MIT** datasets for training and evaluation.

**USPTO-50k** is obtained it from: https://github.com/Hanjun-Dai/GLN, which is in `.csv` format.
**USPTO-full** is directly obtained from: https://github.com/coleygroup/Graph2SMILES, they provided a preprocessed `.txt` data that had already been splited into tokens with a regular regex.
**USPTO-MIT** is obtained from: https://github.com/pschwllr/MolecularTransformer, we used the **mixed** version which was also in `.txt` format.

We provide the `.zip` file for **USPTO-50k** dataset at `model/preprocess/data/raw_csv/$DATASET$`, and we also provide the download script for **USPTO-MIT** and **USPTO-full**, you can directly running the preprocess scripts without downloading them manually:

```
python Model_running.py --request preprocess --dataset 50k  for USPTO-50k
python Model_running.py --request preprocess --dataset MIT  for USPTO-MIT
python Model_running.py --request preprocess --dataset full  for USPTO-full
```

You can also choose to download the datasets manually at https://www.dropbox.com/sh/aa41sxlte7wngiv/AADWe3XEg2C1wBbfz3lktyjFa?dl=0, and then decompress the datasets into `model/preprocess/data/raw_csv/$DATASET$`.

## 4. Training and evaling during training
We provide the hyperparameter settings in `args_settings/`, or you can run the following scripts to train the model in each dataset with the same configuration as we mention in the paper:

```
python Model_running.py --request train --dataset 50k  for USPTO-50k
python Model_running.py --request train --dataset 50k_class  for USPTO-50k with reaction type
python Model_running.py --request train --dataset MIT  for USPTO-MIT
python Model_running.py --request train --dataset full  for USPTO-full
```

You can find the training log and checkpoints at `model/check_point/$DATASET$`.

The automatic weight averaging is available in our model, which can automatically generate weight average models based on weighted accuracy on the validation set. You can acquire multiple weight average models from the different training steps of the model after it has been trained, as well as the original models and the accuracy of each model. Use this information to quickly select the best model checkpoint for subsequent evaluation.

## 5. Evaluation for one-step reaction prediction
We also provide the evaluation hyperparameter settings in `args_settings/`, or you can run the following scripts to evaluate the model in each dataset:

```
python Model_running.py --request test --dataset $DATASET$ --ckpt_list $CHECKPOINT$
```

DATASET means the name of training data, which is one of [50k, 50k_class, MIT, full].
CHECKPOINT means the checkpoint name for evaluation, such as '50k.ckpt'.

You can also run the evaluation with pretrained model checkpoints for each dataset, the scripts are as follow:

```
python Model_running.py --request test --dataset $DATASET$ --download_checkpoint
```

For the model checkpoints for each datasets, you can also download them manually at: https://www.dropbox.com/sh/uekt2uacoawz81g/AABIc4w-pQAWxiYk4geSgMBpa?dl=0, each of the checkpoint need to be pasted into `model/check_point/$DATASET$`.

The evaluation will report the accuracy and invalid rate from top-1 to top-10 by default.

# BiG2S: A Dual Task Graph-to-Sequence Model for the End-to-End Template-Free Reaction Predicting

Here is the code for *"BiG2S: A Dual Task Graph-to-Sequence Model for the End-to-End Template-Free Reaction Predicting"*

***The manuscript is under submission now, the code and checkpoint files will be uploaded later***

## 1. Environment setup
Code was run and evaluated for:

    - python 3.10.8
    - pytorch 1.13.0
    - pytorch-scatter 2.1.0
    - PyG 2.2.0
    - rdkit 2022.09.1

Models were trained on RTX A5000 with 24GB memory for larger batch size(e.g. 128\*2), which also available for less GPU memory with an appropriate batch size setting and larger gradient accumulation steps(e.g. 32\*2 and accumulate 4 steps for 6GB).

Note that an earlier version of rdkit(e.g. 2019) may result in different SMILES canonicalization results.

For more details about the conda environment, please check the `condalist.txt`.

## 2. Data preprocessing
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
Please replace the `args.dataset_name` at the end of the code with the corresponding dataset name in advance(`uspto_50k` is the default, you can choose `uspto_full` or `uspto_MIT`. Please ensure the `args.split_preprocess` is set to `True` when choosing `uspto_full` or `uspto_MIT` for more efficient evaluation
during training).

The datasets mentioned above will be uploaded later for a more convenient data acquirement.

## 3. Training and validation during training
We provide the hyperparameter settings in `args_settings/`.

Choose and copy the settings corresponding to the dataset and paste them into the end of the training script `Module_training.py` to replace the default args setting. When using the appropriate parameters, the model will be trained as we described in the paper by running the following training script:

```
python Module_training.py
```

The automatic weight averaging is available in our model, which can automatically generate weight average models based on weighted accuracy on the validation set. You can acquire multiple weight average models from the different training steps of the model after it has been trained, as well as the original models and the accuracy of each model. Use this information to quickly select the best model checkpoint for subsequent evaluation.

## 4. Evaluation for one-step reaction predicting
We also provide the evaluation hyperparameter settings in `args_settings/`.

Choose and copy the settings corresponding to the dataset and paste them into the end of the evaling script `Module_evaling.py` to replace the default args setting. Please replace `args.ckpt_list` with the checkpoint file name if you choose other checkpoints for evaluation.

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

Now you can find the corresponding top-10 prediction results of these four tasks and the visualization results from RDKit in `tgt_name`.
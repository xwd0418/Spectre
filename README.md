# SPECTRE: A Spectral Transformer for Molecule Identification

## Installing packages:
```bash
pip install -r requirements.txt
```

## Cloning the repo
Please unzip the repo as `/root/MorganFP_prediction/reproduce_previous_works/Spectre` to best reproduce the code


## Dataset Structure
In our repo, we have hard-coded our datasets' name and file paths.

Here `OneD_Only_Dataset` contains compounds where only 1D NMRs are available. `SMILES_dataset` contains compounds where 2D HSQC NMRs are always availabel and 1D NMRs are sometimes available 

The dataset root path in our project is `/workspace`. Our dataset, with HSQC spectra removed due to legal concerns, is available at [here](https://drive.google.com/file/d/16eKPJz2hwsfnPIKweJyKrfi9cZiGVErD/view?usp=sharing).

After downloadind the dataset, please run `unzip -q DatasetWithoutHSQC.zip -d /workspace/`

HYUN_FP refers to Hyunwoo Kim's proposed fingerprint used in previous SOTA method: DeepSAT. MW/index.pkl, Chemical/index.pkl, and SMILES/index.pkl records the molecular weight, chemical name, and SMILES string of each comound.


```bash
./OneD_Only_Dataset:

./OneD_Only_Dataset/test:

./OneD_Only_Dataset/test/Chemical:
./OneD_Only_Dataset/test/Chemical/index.pkl

./OneD_Only_Dataset/test/oneD_NMR:
./OneD_Only_Dataset/test/oneD_NMR/10036.pt
./OneD_Only_Dataset/test/oneD_NMR/100198.pt

./OneD_Only_Dataset/test/SMILES:
./OneD_Only_Dataset/test/SMILES/index.pkl

./OneD_Only_Dataset/test/HYUN_FP:
./OneD_Only_Dataset/test/HYUN_FP/102924.pt
./OneD_Only_Dataset/test/HYUN_FP/102936.pt

./OneD_Only_Dataset/test/MW:
./OneD_Only_Dataset/test/MW/index.pkl

./OneD_Only_Dataset/test/Isomeric_SMILES:
./OneD_Only_Dataset/test/Isomeric_SMILES/index.pkl

./OneD_Only_Dataset/train:

./OneD_Only_Dataset/train/Chemical:
./OneD_Only_Dataset/train/Chemical/index.pkl

./OneD_Only_Dataset/train/oneD_NMR:
./OneD_Only_Dataset/train/oneD_NMR/0.pt
./OneD_Only_Dataset/train/oneD_NMR/100004.pt

./OneD_Only_Dataset/train/SMILES:
./OneD_Only_Dataset/train/SMILES/index.pkl

./OneD_Only_Dataset/train/HYUN_FP:
./OneD_Only_Dataset/train/HYUN_FP/100001.pt
./OneD_Only_Dataset/train/HYUN_FP/100002.pt

./OneD_Only_Dataset/train/MW:
./OneD_Only_Dataset/train/MW/index.pkl

./OneD_Only_Dataset/train/Isomeric_SMILES:
./OneD_Only_Dataset/train/Isomeric_SMILES/index.pkl

./OneD_Only_Dataset/val:

./OneD_Only_Dataset/val/Chemical:
./OneD_Only_Dataset/val/Chemical/index.pkl

./OneD_Only_Dataset/val/oneD_NMR:
./OneD_Only_Dataset/val/oneD_NMR/10000.pt
./OneD_Only_Dataset/val/oneD_NMR/100014.pt

./OneD_Only_Dataset/val/SMILES:
./OneD_Only_Dataset/val/SMILES/index.pkl

./OneD_Only_Dataset/val/HYUN_FP:
./OneD_Only_Dataset/val/HYUN_FP/10000.pt
./OneD_Only_Dataset/val/HYUN_FP/100003.pt

./OneD_Only_Dataset/val/MW:
./OneD_Only_Dataset/val/MW/index.pkl

./OneD_Only_Dataset/val/Isomeric_SMILES:
./OneD_Only_Dataset/val/Isomeric_SMILES/index.pkl

./SMILES_dataset:

./SMILES_dataset/train:

./SMILES_dataset/train/MW:
./SMILES_dataset/train/MW/index.pkl

./SMILES_dataset/train/HSQC:
./SMILES_dataset/train/HSQC/5823.pt
./SMILES_dataset/train/HSQC/105818.pt

./SMILES_dataset/train/HYUN_FP:
./SMILES_dataset/train/HYUN_FP/5273.pt
./SMILES_dataset/train/HYUN_FP/108632.pt

./SMILES_dataset/train/Chemical:
./SMILES_dataset/train/Chemical/index.pkl

./SMILES_dataset/train/SMILES:
./SMILES_dataset/train/SMILES/index.pkl

./SMILES_dataset/train/oneD_NMR:
./SMILES_dataset/train/oneD_NMR/100.pt
./SMILES_dataset/train/oneD_NMR/1000.pt

./SMILES_dataset/train/Isomeric_SMILES:

./SMILES_dataset/val:

./SMILES_dataset/val/MW:
./SMILES_dataset/val/MW/index.pkl

./SMILES_dataset/val/HSQC:
./SMILES_dataset/val/HSQC/1425.pt
./SMILES_dataset/val/HSQC/457.pt

./SMILES_dataset/val/HYUN_FP:
./SMILES_dataset/val/HYUN_FP/1425.pt
./SMILES_dataset/val/HYUN_FP/457.pt

./SMILES_dataset/val/Chemical:
./SMILES_dataset/val/Chemical/index.pkl

./SMILES_dataset/val/SMILES:
./SMILES_dataset/val/SMILES/index.pkl

./SMILES_dataset/val/oneD_NMR:
./SMILES_dataset/val/oneD_NMR/1000.pt
./SMILES_dataset/val/oneD_NMR/10034.pt

./SMILES_dataset/val/Isomeric_SMILES:

./SMILES_dataset/test:

./SMILES_dataset/test/MW:
./SMILES_dataset/test/MW/index.pkl

./SMILES_dataset/test/HSQC:
./SMILES_dataset/test/HSQC/6777.pt
./SMILES_dataset/test/HSQC/635.pt

./SMILES_dataset/test/HYUN_FP:
./SMILES_dataset/test/HYUN_FP/6777.pt
./SMILES_dataset/test/HYUN_FP/635.pt

./SMILES_dataset/test/Chemical:
./SMILES_dataset/test/Chemical/index.pkl

./SMILES_dataset/test/SMILES:
./SMILES_dataset/test/SMILES/index.pkl

./SMILES_dataset/test/oneD_NMR:
./SMILES_dataset/test/oneD_NMR/10002.pt
./SMILES_dataset/test/oneD_NMR/10013.pt

./SMILES_dataset/test/Isomeric_SMILES:
```

## Data Processing
Execute these scripts to pre-built necessary pickle files for training

`/root/MorganFP_prediction/reproduce_previous_works/Spectre/notebooks/dataset_building/find_all_info_indices.ipynb`

`/root/MorganFP_prediction/reproduce_previous_works/Spectre/notebooks/dataset_building/generate_all_morganFP.ipynb`


## A brief explanation of the files 

- datasets:

  - hsqc_folder_dataset.py: A dataset file to load a given combination of NMRs, such as only HSQC, HSQC and H-NMR, all three NMRs, etc
  - oneD_dataset.py: A dataset file to load a given combination of NMRs, when all the NMRs are 1d NMR, i.e.,only H-NMR, only C-NMR, and only both 1d NMRs
  - optional_2d_folder_dataset: It is able to load all possible combinations of NMRs, and it is used to train models taking optional inputs. 

- models:

  - ranked_transformer.py : A Transformer model that can take a specific NMR combination
  - ranked_resnet.py: A Resnet model that can take a specific NMR combination
  - optional_input_ranked_transformer.py: A Transformer model that can take any NMR combination


### Running

To train a model that can take any input NMR combination to predict entropy-based-fingerprint, specifically using r0-r4 entropy-based Morgan fingerprint as target
- ` python train_ranker_transformer.py transformer_2d1d --foldername flexible_models_best_FP  --expname r0_r4_FP_trial_3 --optional_inputs true --combine_oneD_only_dataset true --random_seed 303 --FP_choice pick_entropy_r4 --scheduler attention --wavelength_bounds 0.01 400.0 --wavelength_bounds 0.01 20.0 ` 

To train a model that takes only HSQC, using r0-r5 entropy-based Morgan fingerprint:
- `python train_ranker_transformer.py transformer_2d1d --foldername train_on_all_data_possible --random_seed 101 --expname only_hsqc_trial_1 --use_oneD_NMR_no_solvent false  --FP_choice pick_entropy_r3 --scheduler attention  --wavelength_bounds 0.01 400.0 --wavelength_bounds 0.01 20.0 `

To train a model that takes only H-NMR, using r0-r3 entropy-based Morgan fingerprint:
- `python train_ranker_transformer.py transformer_2d1d --foldername train_on_all_data_possible --random_seed 101 --only_H_NMR true --only_oneD_NMR true --combine_oneD_only_dataset true --expname only_h_trial_1   --FP_choice pick_entropy_r4 --scheduler attention  --wavelength_bounds 0.01 400.0 --wavelength_bounds 0.01 20.0  `

To train a model that takes only C-NMR, using r0-r2 entropy-based Morgan fingerprint:
- `python train_ranker_transformer.py transformer_2d1d --foldername train_on_all_data_possible --random_seed 101 --only_C_NMR true --only_oneD_NMR true --combine_oneD_only_dataset true --expname only_c_trial_1   --FP_choice pick_entropy_r3 --scheduler attention  --wavelength_bounds 0.01 400.0 --wavelength_bounds 0.01 20.0  `

To train a model that takes HSQC and C-NMR, using r0-r2 entropy-based Morgan fingerprint:
- `python train_ranker_transformer.py transformer_2d1d  --foldername train_on_all_data_possible --random_seed 101 --expname HSQC_and_C_trial_1 --only_C_NMR true  --FP_choice pick_entropy_r3 --scheduler attention  --wavelength_bounds 0.01 400.0 --wavelength_bounds 0.01 20.0 `

To train a model that takes HSQC and H-NMR, using r0-r2 entropy-based Morgan fingerprint:
- `python train_ranker_transformer.py transformer_2d1d  --foldername train_on_all_data_possible --random_seed 101 --expname HSQC_and_H_trial_1 --only_H_NMR true  --FP_choice pick_entropy_r4 --scheduler attention  --wavelength_bounds 0.01 400.0 --wavelength_bounds 0.01 20.0`

To train a model that takes C-NMR and H-NMR, using r0-r2 entropy-based Morgan fingerprint:
- `python train_ranker_transformer.py transformer_2d1d --train_on_all_info_set true --foldername entropy_radius_exps_1d --random_seed 101 --only_oneD_NMR true --expname R0_to_R13_only_1d_trial_1   --FP_choice pick_entropy_r2 --scheduler attention  --wavelength_bounds 0.01 400.0 --wavelength_bounds 0.01 20.0`

To train a model that takes all three kinds of NMRs, using r0-r2 entropy-based Morgan fingerprint:
- `python train_ranker_transformer.py transformer_2d1d --train_on_all_info_set true --foldername entropy_radius_exps_all_info --random_seed 101 --expname R0_to_R8_all_info_trial_1  --FP_choice pick_entropy_r2 --scheduler attention  --wavelength_bounds 0.01 400.0 --wavelength_bounds 0.01 20.0  `

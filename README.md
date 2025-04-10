# SPECTRE: A Spectral Transformer for Molecule Identification

## Installing packages:
```bash
pip install -r requirements.txt
```

## It is more encouraged to directly use my docker image:
gitlab-registry.nrp-nautilus.io/w6xu/guru-docker-image:3699f97c

## Dataset Structure
Our dataset, with HSQC spectra removed due to legal concerns, is available at [here](https://drive.google.com/file/d/16eKPJz2hwsfnPIKweJyKrfi9cZiGVErD/view?usp=sharing).

Unfortunately, in our repo, we have hard-coded our datasets' name and file paths. Everyone is more than welcome to orgnize the paths. Otherwise, we suggest to 
- Unzip the dataset zip at /workspace  `unzip -q DatasetWithoutHSQC.zip -d /workspace/`

- Git clone our Spectre repo at `/root/gurusmart/MorganFP_prediction/reproduce_previous_works`

Here `OneD_Only_Dataset` contains compounds where only 1D NMRs are available. `SMILES_dataset` contains compounds where 2D HSQC NMRs are always availabel and 1D NMRs are sometimes available 


HYUN_FP refers to Hyunwoo Kim's proposed fingerprint used in previous SOTA method: DeepSAT. MW/index.pkl, Chemical/index.pkl, and SMILES/index.pkl records the molecular weight, chemical name, and SMILES string of each comound.



## Data Processing before training the models
### (No longer required) Execute these scripts to pre-built pickle files if you want to get fast access to the molecules that have all three types of NMRs (HSQC, C-NMR, H-NMR) 

`{repo_path}/Spectre/notebooks/dataset_building/find_all_info_indices.ipynb`

`{repo_path}/Spectre/notebooks/dataset_building/generate_all_morganFP.ipynb`

### Analyze the entropy of each fragments in the dataset (this step is not needed if you can download our inside version of dataset)
`bash {repo_path}/Spectre/notebook_and_scripts/SMILES_fragmenting/build_dataset_specific_FP/save_zip.sh`

### Build retrieval set of more than 500K molecules 
Download the pkl file including their SMILES, Names, Molecular Weights

- Download the pickle file from [goolge-drive](https://drive.google.com/file/d/1Vh_oWVYhRg2h0-y2E8NI5pVTNgYKMFCj/view?usp=drive_link) and save as `/root/gurusmart/MorganFP_prediction/inference_data/coconut_loutus_hyun_training/inference_metadata_latest_RDkit.pkl`

- Run this script to generate fingerprint of the retrieval sets (by default we use radius from 0 to 6 and build fingperprint of length 16384):
`step=1 python {repo_path}/Spectre/notebook_and_scripts/dataset_building/build_infernce_set_db_specific.py`
`step=2 python {repo_path}/Spectre/notebook_and_scripts/dataset_building/build_infernce_set_db_specific.py`

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

To train a model that can take any input NMR combination to predict entropy-based-fingerprint:
` python train_ranker_transformer.py transformer_2d1d --foldername flexible_models_jittering_size_1  --expname r0_r6 --optional_inputs true --combine_oneD_only_dataset true --random_seed 1 --FP_choice Hash_Entropy_FP_R_6  --out_dim 16384 --jittering 1 ` 

- If you want to make "Molecular Weight" also an optional input, add an extra flag `--optional_MW true`

If you want to train a non-flexible model, i.e., only accecpt fixed type of NMR(s), you can refer this [github page](https://github.com/xwd0418/Guru-research_configs/blob/main/jobs/db_specific_entropy_based/run_all_possible_input_jitter.sh) to see how I schedule the training in different settings.

For example, to train a model that takes only HSQC:
`python train_ranker_transformer.py transformer_2d1d --foldername train_on_all_data_possible_with_jittering --jittering 1 --random_seed 1 --expname only_hsqc --use_oneD_NMR_no_solvent false  --FP_choice Hash_Entropy_FP_R_6  --out_dim 16384 --jittering 1  `


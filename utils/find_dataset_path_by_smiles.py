import pickle, os
from collections import defaultdict

# all_smiles_train_2d = pickle.load(open('/workspace/SMILES_dataset/train/SMILES/index.pkl', 'rb'))
# all_smiles_val_2d = pickle.load(open('/workspace/SMILES_dataset/val/SMILES/index.pkl', 'rb'))
# all_smiles_test_2d = pickle.load(open('/workspace/SMILES_dataset/test/SMILES/index.pkl', 'rb'))

# all_chemical_names_train_2d = pickle.load(open('/workspace/SMILES_dataset/train/Chemical/index.pkl', 'rb'))
# all_chemical_names_val_2d = pickle.load(open('/workspace/SMILES_dataset/val/Chemical/index.pkl', 'rb'))
# all_chemical_names_test_2d = pickle.load(open('/workspace/SMILES_dataset/test/Chemical/index.pkl', 'rb'))

# all_smiles_train_1d = pickle.load(open('/workspace/OneD_Only_Dataset/train/SMILES/index.pkl', 'rb'))
# all_smiles_val_1d = pickle.load(open('/workspace/OneD_Only_Dataset/val/SMILES/index.pkl', 'rb'))
# all_smiles_test_1d = pickle.load(open('/workspace/OneD_Only_Dataset/test/SMILES/index.pkl', 'rb'))

# all_chemical_names_train_1d = pickle.load(open('/workspace/OneD_Only_Dataset/train/Chemical/index.pkl', 'rb'))
# all_chemical_names_val_1d = pickle.load(open('/workspace/OneD_Only_Dataset/val/Chemical/index.pkl', 'rb'))
# all_chemical_names_test_1d = pickle.load(open('/workspace/OneD_Only_Dataset/test/Chemical/index.pkl', 'rb'))
with open('/workspace/SMILES_dataset/train/SMILES/index.pkl', 'rb') as f:
    all_smiles_train_2d = pickle.load(f)    
with open('/workspace/SMILES_dataset/val/SMILES/index.pkl', 'rb') as f:
    all_smiles_val_2d = pickle.load(f)
with open('/workspace/SMILES_dataset/test/SMILES/index.pkl', 'rb') as f:
    all_smiles_test_2d = pickle.load(f)
with open('/workspace/SMILES_dataset/train/Chemical/index.pkl', 'rb') as f:
    all_chemical_names_train_2d = pickle.load(f)
with open('/workspace/SMILES_dataset/val/Chemical/index.pkl', 'rb') as f:
    all_chemical_names_val_2d = pickle.load(f)
with open('/workspace/SMILES_dataset/test/Chemical/index.pkl', 'rb') as f:
    all_chemical_names_test_2d = pickle.load(f)
with open('/workspace/OneD_Only_Dataset/train/SMILES/index.pkl', 'rb') as f:
    all_smiles_train_1d = pickle.load(f)
with open('/workspace/OneD_Only_Dataset/val/SMILES/index.pkl', 'rb') as f:
    all_smiles_val_1d = pickle.load(f)
with open('/workspace/OneD_Only_Dataset/test/SMILES/index.pkl', 'rb') as f:
    all_smiles_test_1d = pickle.load(f)
with open('/workspace/OneD_Only_Dataset/train/Chemical/index.pkl', 'rb') as f:
    all_chemical_names_train_1d = pickle.load(f)
with open('/workspace/OneD_Only_Dataset/val/Chemical/index.pkl', 'rb') as f:
    all_chemical_names_val_1d = pickle.load(f)
with open('/workspace/OneD_Only_Dataset/test/Chemical/index.pkl', 'rb') as f:
    all_chemical_names_test_1d = pickle.load(f)

identifier ={
    "2d": "HSQC",
    "1d": "oneD_NMR/"
}
dataset_root_dir ={
    "2d": "/workspace/SMILES_dataset",
    "1d": "/workspace/OneD_Only_Dataset"
}
smiles_to_path = defaultdict(list)
for datasrc in ['2d',"1d" ]:
    for split in ['train', 'val', 'test']:
        all_smiles = locals()['all_smiles_' + split + '_' + datasrc]
        # all_names = locals()['all_chemical_names_' + split + '_' + datasrc]
        for f in os.listdir(os.path.join(dataset_root_dir[datasrc], split, identifier[datasrc])):
            curr_smiles = all_smiles[int(f.split(".")[0])]
            curr_parent_dir, curr_file = os.path.join(dataset_root_dir[datasrc], split), f.split(".")[0]
            smiles_to_path[curr_smiles].append((curr_parent_dir, curr_file))
            
def find_dataset_path_by_smiles(smiles):
    return smiles_to_path[smiles]

/opt/conda/bin/python /root/gurusmart/MorganFP_prediction/reproduce_previous_works/Spectre/notebooks/SMILES_fragmenting/build_dataset_specific_FP/find_most_frequent_frags.py

cd /workspace && zip -r Spectre_all_frag_DB_Count_exists.zip SMILES_dataset OneD_Only_Dataset  *_radius_under_10.pkl 
cp Spectre_all_frag_DB_Count_exists.zip /root/gurusmart/MorganFP_prediction/entropy_based_datasets 
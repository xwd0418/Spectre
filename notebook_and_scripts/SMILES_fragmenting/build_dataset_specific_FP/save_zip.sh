# python /root/gurusmart/MorganFP_prediction/reproduce_previous_works/Spectre/notebook_and_scripts/SMILES_fragmenting/build_dataset_specific_FP/find_frags.py

cd /workspace 
zip -r entropy_of_hashes_DB.zip SMILES_dataset OneD_Only_Dataset  *under_radius_10.pkl 
cp entropy_of_hashes_DB.zip /root/gurusmart/MorganFP_prediction/entropy_based_datasets 

# zip -u /root/gurusmart/MorganFP_prediction/entropy_based_datasets/entropy_of_hashes_DB.zip  *under_radius_10.pkl

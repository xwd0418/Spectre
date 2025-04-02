python /root/gurusmart/MorganFP_prediction/reproduce_previous_works/Spectre/notebook_and_scripts/SMILES_fragmenting/build_dataset_specific_FP/find_frags.py

cd /workspace 
# zip -r Spectre_all_frag_DB_Count_exists.zip SMILES_dataset OneD_Only_Dataset  *_radius_under_10.pkl 
# cp Spectre_all_frag_DB_Count_exists.zip /root/gurusmart/MorganFP_prediction/entropy_based_datasets 

zip -u /root/gurusmart/MorganFP_prediction/entropy_based_datasets/Spectre_all_frag_DB_Count_testset_NP_classify.zip  *_radius_under_10.pkl

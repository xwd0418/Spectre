echo "copying"
zipname=entropy_of_hashes_DB.zip
# zipname=Spectre_all_frag_DB_Count_testset_NP_classify.zip
# zipname=Spectre_2d1d_DB_specific_FP.zip
# zipname=combined_two_datasets_and_rankingsets.zip
# zipname=RemoveIsomericInfoSMILES.zip
# zipname=Spectre_2d1d_canonical_smiles_wo_tautomer.zip   # too many 1d: 200k+
cp /root/gurusmart/MorganFP_prediction/entropy_based_datasets/$zipname  /workspace/

echo "start to unzip..."

# unzip -q /root/gurusmart/MorganFP_prediction/entropy_based_datasets/combined_two_datasetes_and_rankingsets_without_empty_oned_NMR.zip -d /workspace/
# unzip -q /workspace/combined_two_datasetes_and_rankingsets_without_empty_oned_NMR.zip -d /workspace/
# unzip -q /workspace/weird_H_and_tautomer_cleaned.zip -d /workspace/
# unzip -q /workspace/RemoveIsomericInfoSMILES.zip -d /workspace/
unzip /workspace/$zipname -d /workspace/

# SMART_2D_combined_by_canonical_smiles.zip
# OneD_Only_Dataset.zip


# unzip -q /root/gurusmart/MorganFP_prediction/cleaned_dataset/ranking_sets_with_all_info_molecules.zip -d /workspace

echo "unzip done"

# zip -r combined.zip folder1 folder2

# unzip  /root/gurusmart/MorganFP_prediction/cleaned_dataset/SMART_1d2d_cleaned_by_InChi.zip -d /
# unzip  /root/gurusmart/MorganFP_prediction/cleaned_dataset/ranking_sets_cleaned_by_InChi.zip -d /

# if [ "$1" == "old" ]; then
#     echo "old dataset"
#     unzip -q /root/gurusmart/MorganFP_prediction/James_dataset_zips/Smart_NPMRD_dataset.zip -d /workspace/
#     unzip -q /root/gurusmart/MorganFP_prediction/reproduce_previous_works/smart4.5/ranking_sets_hyun_and_r2.zip -d /workspace
# else
#     echo "new dataset"
#     unzip -q /root/gurusmart/MorganFP_prediction/cleaned_dataset/SMART_Additional_FPs.zip -d /workspace/
#     unzip -q /root/gurusmart/MorganFP_prediction/cleaned_dataset/ranking_sets_with_additional_FP.zip -d /workspace
# fi

# just FPr0-r4 
# unzip -q /root/gurusmart/MorganFP_prediction/cleaned_dataset/SMART_Additional_FPs.zip -d /workspace/
# unzip -q /root/gurusmart/MorganFP_prediction/cleaned_dataset/ranking_sets_with_additional_FP.zip -d /workspace

# FP from r2-r10



if [ "$1" == "more" ]; then
    echo "r2-r10 dataset"
    unzip -q /root/gurusmart/MorganFP_prediction/cleaned_dataset/SMART_FPs_from_r2_to_r10.zip -d /workspace/
    # unzip -q /root/gurusmart/MorganFP_prediction/cleaned_dataset/ranking_FPs_from_r2_to_r10.zip -d /workspace

elif [ "$1" == "cnn" ]; then
    echo "dataset cleaned with image HSQCs"
    unzip -q /root/gurusmart/MorganFP_prediction/cleaned_dataset/SMART_image_HSQC.zip -d /workspace/


else
    echo "dataset cleaned by inchi, with r0-r4, and count-based. Entropy based FP is exact radius"
    unzip -q /root/gurusmart/MorganFP_prediction/cleaned_dataset/SMART_FPs_exact_radii.zip -d /workspace/
    # SMART_2D_combined_by_canonical_smiles.zip
    # OneD_Only_Dataset.zip
fi

unzip -q /root/gurusmart/MorganFP_prediction/cleaned_dataset/ranking_sets_with_all_info_molecules.zip -d /workspace

echo "unzip done"
python train_concise.py hsqc_transformer --foldername new_split --expname "tightbound-384" --ranking_set_path \
  "/workspace/smart4.5/tempdata/SMILES_ranking_sets/val/rankingset.pt" \
  --dim_model 384 --dim_coords "180,180,24" --wavelength_bounds 0.01 400 --wavelength_bounds 0.01 20 \
  --coord_enc "sce"
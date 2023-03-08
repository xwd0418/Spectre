python train_concise.py hsqc_transformer --foldername new_split --expname "h8-l4-256" --ranking_set_path \
  "/workspace/smart4.5/tempdata/SMILES_ranking_sets/val/rankingset.pt" \
  --dim_model 256 --dim_coords "120,120,16" --wavelength_bounds 0.01 400 --wavelength_bounds 0.01 20 \
  --coord_enc "sce" --heads 8 --layers 4
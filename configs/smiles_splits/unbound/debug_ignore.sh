python train_concise.py hsqc_transformer --foldername new_split --expname debug --ranking_set_path \
  "/workspace/smart4.5/tempdata/SMILES_ranking_sets/val/rankingset.pt" --data_len 200 --epochs 4 \
  --dim_model 128 --dim_coords "60,60,8" --wavelength_bounds 0.01 400 --wavelength_bounds 0.01 20 \
  --coord_enc "sce"
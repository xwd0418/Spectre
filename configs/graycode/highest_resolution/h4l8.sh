python train_concise.py hsqc_transformer --foldername graycode --expname "h4-l8-256-r0.01" --ranking_set_path \
  "/workspace/smart4.5/tempdata/SMILES_ranking_sets/val/rankingset.pt" \
  --dim_model 256 --dim_coords "120,120,16" \
  --coord_enc "gce" --gce_resolution 0.01 --heads 4 --layers 8
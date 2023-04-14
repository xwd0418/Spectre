python train_concise.py hsqc_transformer --foldername graycode --expname "h8-l4-256-r0.1" --ranking_set_path \
  "/workspace/smart4.5/tempdata/SMILES_ranking_sets/val/rankingset.pt" \
  --dim_model 256 --dim_coords "120,120,16" \
  --coord_enc "gce" --gce_resolution 0.1 --heads 8 --layers 4
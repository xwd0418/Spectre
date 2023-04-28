python train_concise.py hsqc_transformer --foldername new_split --expname debug --force_start True --ranking_set_path \
  "/workspace/smart4.5/tempdata/SMILES_ranking_sets/val/rankingset.pt" --data_len 200 --epochs 4 \
  --dim_model 128 --dim_coords "60,60,8" \
  --coord_enc "gce" --gce_resolution 0.1
python ./train_synshapes_structurenet.py \
  --exp_name 'exp_structurenet_vae_synshapes' \
  --category 'SynChair' \
  --train_dataset 'train.txt' \
  --val_dataset 'test_small.txt' \
  --epochs 300 \
  --model_version 'model_structurenet'

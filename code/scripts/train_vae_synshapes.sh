python ./train_synshapes.py \
  --model_version 'model' \
  --exp_name "exp_vae_synshapes" \
  --category 'SynChair' \
  --train_dataset 'train.txt' \
  --epochs 300 \
  --shapediff_topk 96

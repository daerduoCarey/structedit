python ./train.py \
  --model_version 'model' \
  --exp_name 'exp_ae_chair_sd' \
  --category 'Chair' \
  --data_path '../data/partnetdata/chair_hier' \
  --train_dataset 'train_no_other_less_than_10_parts.txt' \
  --epochs 1000 \
  --shapediff_metric 'sd' \
  --non_variational

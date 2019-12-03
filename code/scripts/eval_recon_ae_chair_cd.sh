python eval_recon.py \
  --exp_name 'exp_ae_chair_cd' \
  --test_dataset 'test_no_other_less_than_10_parts.txt' \
  --baseline_dir '../data/results/exp_structurenet_ae_chair_recon' \
  --shapediff_topk 20 \
  --start_id 0 \
  --end_id -1

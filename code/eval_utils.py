import os
import sys
import numpy as np

def compute_recon_numbers(in_dir, baseline_dir, shapediff_topk):
    topk_cd = np.zeros((shapediff_topk), dtype=np.float32)
    topk_sd = np.zeros((shapediff_topk), dtype=np.float32)
    baseline_topk_cd = np.zeros((shapediff_topk), dtype=np.float32)
    baseline_topk_sd = np.zeros((shapediff_topk), dtype=np.float32)
    topk_cnt = np.zeros((shapediff_topk), dtype=np.int32)
    for anno_id in os.listdir(in_dir):
        if '.' not in anno_id:
            cur_dir = os.path.join(in_dir, anno_id)
            for item in os.listdir(cur_dir):
                if item.endswith('.stats'):
                    nid = int(item.split('.')[0].split('_')[1])
                    neighbor_anno_id = item.split('.')[0].split('_')[2]
                    with open(os.path.join(cur_dir, item), 'r') as fin:
                        topk_cd[nid] += float(fin.readline().rstrip().split()[-1])
                        topk_sd[nid] += float(fin.readline().rstrip().split()[-1])
                    with open(os.path.join(baseline_dir, neighbor_anno_id, 'stats.txt'), 'r') as fin:
                        baseline_topk_cd[nid] += float(fin.readline().rstrip().split()[-1])
                        baseline_topk_sd[nid] += float(fin.readline().rstrip().split()[-1])
                    topk_cnt[nid] += 1

    topk_cd /= topk_cnt
    topk_sd /= topk_cnt
    baseline_topk_cd /= topk_cnt
    baseline_topk_sd /= topk_cnt
    print('ours cd mean: %.5f' % np.mean(topk_cd))
    print('ours sd mean: %.5f' % np.mean(topk_sd))
    print('structurenet cd mean: %.5f' % np.mean(baseline_topk_cd))
    print('structurenet sd mean: %.5f' % np.mean(baseline_topk_sd))

    with open(os.path.join(in_dir, 'stats.txt'), 'w') as fout:
        fout.write('ours cd mean: %.5f\n' % np.mean(topk_cd))
        fout.write('ours sd mean: %.5f\n' % np.mean(topk_sd))
        fout.write('structurenet cd mean: %.5f\n' % np.mean(baseline_topk_cd))
        fout.write('structurenet sd mean: %.5f\n' % np.mean(baseline_topk_sd))
        for i in range(shapediff_topk):
            fout.write('%d %d %.5f %.5f %.5f %.5f\n' % (i, topk_cnt[i], topk_cd[i], topk_sd[i], baseline_topk_cd[i], baseline_topk_sd[i]))



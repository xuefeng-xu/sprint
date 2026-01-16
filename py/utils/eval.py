import numpy as np


def mean_n_recall_at_k(pred, gt, n, k):
    recalls = []
    for pred_i, gt_i in zip(pred, gt):
        if len(pred_i) > n:
            pred_i = pred_i[:n]
        if len(gt_i) > k:
            gt_i = gt_i[:k]

        pred_i = set(pred_i)
        gt_i = set(gt_i)

        # skip if gt is empty
        if not gt_i:
            continue

        hits = len(gt_i & pred_i)
        recalls.append(hits / len(gt_i))

    return float(np.mean(recalls)) if recalls else 0.0

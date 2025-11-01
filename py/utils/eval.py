import numpy as np


def mean_recall_at_k(pred, gt, k):
    recalls = []
    for pred_i, gt_i in zip(pred, gt):
        if len(pred_i) > k:
            pred_i = pred_i[:k]
        if len(gt_i) > k:
            gt_i = gt_i[:k]

        pred_i = set(pred_i)
        gt_i = set(gt_i)

        if not gt_i:
            continue  # skip if gt is empty

        hits = len(gt_i & pred_i)
        recalls.append(hits / len(gt_i))

    return np.mean(recalls) if recalls else 0.0

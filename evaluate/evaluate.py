import numpy as np
from sklearn.metrics.cluster import (contingency_matrix,
                                     normalized_mutual_info_score)
from sklearn.metrics import (precision_score, recall_score)

def pairwise(gt_labels, pred_labels, sparse=True):
    print(gt_labels.shape)
    print(pred_labels.shape)
    n_samples, = gt_labels.shape
    c = contingency_matrix(gt_labels, pred_labels, sparse=sparse)
    tk = np.dot(c.data, c.data) - n_samples
    pk = np.sum(np.asarray(c.sum(axis=0)).ravel()**2) - n_samples
    qk = np.sum(np.asarray(c.sum(axis=1)).ravel()**2) - n_samples

    avg_pre = tk / pk
    avg_rec = tk / qk
    fscore = _compute_fscore(avg_pre, avg_rec)

    return avg_pre, avg_rec, fscore


def _get_lb2idxs(labels):
    lb2idxs = {}
    for idx, lb in enumerate(labels):
        if lb not in lb2idxs:
            lb2idxs[lb] = []
        lb2idxs[lb].append(idx)
    return lb2idxs


def _compute_fscore(pre, rec):
    return 2. * pre * rec / (pre + rec)


def bcubed(gt_labels, pred_labels):
    gt_labels = gt_labels.numpy()
    gt_lb2idxs = _get_lb2idxs(gt_labels)
    pred_lb2idxs = _get_lb2idxs(pred_labels)

    num_lbs = len(gt_lb2idxs)
    pre = np.zeros(num_lbs)
    rec = np.zeros(num_lbs)
    gt_num = np.zeros(num_lbs)

    for i, gt_idxs in enumerate(gt_lb2idxs.values()):
        all_pred_lbs = np.unique(pred_labels[gt_idxs])
        gt_num[i] = len(gt_idxs)
        for pred_lb in all_pred_lbs:
            pred_idxs = pred_lb2idxs[pred_lb]
            n = 1. * np.intersect1d(gt_idxs, pred_idxs).size
            pre[i] += n**2 / len(pred_idxs)
            rec[i] += n**2 / gt_num[i]

    gt_num = gt_num.sum()
    avg_pre = pre.sum() / gt_num
    avg_rec = rec.sum() / gt_num
    fscore = _compute_fscore(avg_pre, avg_rec)

    return avg_pre, avg_rec, fscore


def nmi(gt_labels, pred_labels):
    return normalized_mutual_info_score(pred_labels, gt_labels)


def evaluate(gt_labels, pred_labels, metric='pairwise'):
    if metric == 'pairwise':
        return pairwise(gt_labels, pred_labels)
    elif metric == 'bcubed':
        return bcubed(gt_labels, pred_labels)
    elif metric == 'nmi':
        return nmi(gt_labels, pred_labels)
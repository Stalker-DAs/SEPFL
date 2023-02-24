import numpy as np
import random
import faiss
import torch

from utils import Timer, build_knn, knn2mat
import torch.utils.data as data


def cal_confidence(sim, knn, threshold_rho):
    conf = np.zeros((sim.shape[0],), dtype=np.float32)
    for i, (k_out, s) in enumerate(zip(knn, sim)):
        sum = 0
        for j, k_in in enumerate(k_out):
            sum += 1 - s[j]
        conf[i] = sum
        if i % 1000 == 0:
            print(str(i) + " Finish!")
    conf /= np.abs(conf).max()

    sim = 1 - sim
    over_num = (sim >= threshold_rho)
    sum_num = np.sum(np.array(over_num), axis=1)
    sum_num = sum_num / sum_num.max()
    conf = (conf+sum_num)/2.0

    return conf


def read_meta(fn_meta, start_pos=0, verbose=True):
    lb2idxs = {}
    idx2lb = {}
    with open(fn_meta) as f:
        for idx, x in enumerate(f.readlines()[start_pos:]):
            lb = int(x.strip())
            if lb not in lb2idxs:
                lb2idxs[lb] = []
            lb2idxs[lb] += [idx]
            idx2lb[idx] = lb

    inst_num = len(idx2lb)
    cls_num = len(lb2idxs)
    if verbose:
        print('[{}] #cls: {}, #inst: {}'.format(fn_meta, cls_num, inst_num))
    return lb2idxs, idx2lb


def intdict2ndarray(d, default_val=-1):
    arr = np.zeros(len(d)) + default_val
    for k, v in d.items():
        arr[k] = v
    return arr


def read_probs(path, inst_num, feat_dim, dtype=np.float32, verbose=False):
    assert (inst_num > 0 or inst_num == -1) and feat_dim > 0
    count = -1
    if inst_num > 0:
        count = inst_num * feat_dim
    probs = np.fromfile(path, dtype=dtype, count=count)
    if feat_dim > 1:
        probs = probs.reshape(inst_num, feat_dim)
    if verbose:
        print('[{}] shape: {}'.format(path, probs.shape))
    return probs


def knns2ordered_nbrs(knns, sort=True):
    if isinstance(knns, list):
        knns = np.array(knns)
    nbrs = knns[:, 0, :].astype(np.int32)
    dists = knns[:, 1, :]
    if sort:
        # sort dists from low to high
        nb_idx = np.argsort(dists, axis=1)
        idxs = np.arange(nb_idx.shape[0]).reshape(-1, 1)
        dists = dists[idxs, nb_idx]
        nbrs = nbrs[idxs, nb_idx]
    return dists, nbrs

def classify(conf, rate):
    sum_node = len(conf)

    index = np.zeros(shape=(sum_node)).astype(int)

    sort_conf = torch.argsort(torch.tensor(conf))[:int(sum_node*rate)]
    index[sort_conf] = 1
    return index

def l2norm(vec):
    vec /= np.linalg.norm(vec, axis=1).reshape(-1, 1)
    return vec


class GCNDataset(data.Dataset):
    def __init__(self, cfg):
        self.feat_path = cfg.data['feat_path']
        self.label_path = cfg.data['label_path']
        self.knn_path = cfg.data['knn_path']
        self.feature_dim = cfg.data['feature_dim']
        self.is_sort_knns = True
        self.threshold_rho = cfg.data['threshold_rho']
        self.k = cfg.data['k']
        self.phase = cfg.data['phase']
        self.cf = cfg.data['cf']

        with Timer('load training data'):
            self.lb2idxs, self.idx2lb = read_meta(self.label_path)
            self.inst_num = len(self.idx2lb)
            self.label = intdict2ndarray(self.idx2lb).astype(np.int64)
            self.ignore_label = False

            self.feature = read_probs(self.feat_path, self.inst_num, self.feature_dim)

            self.feature = l2norm(self.feature)

            self.cls_num = len(self.lb2idxs)
            knns = np.load(self.knn_path)['data']
            self.sim, self.knn = knns2ordered_nbrs(knns, sort=self.is_sort_knns)

        with Timer('prepare training data'):
            # calculate confidence
            if cfg.phase == 'train':
                self.conf = cal_confidence(self.sim, self.knn, self.threshold_rho)
            else:
                self.conf = cal_confidence(self.sim, self.knn, self.threshold_rho)


            self.nodes_edge = []

            if self.phase == "train":
                for idx in range(self.inst_num):
                    neighbour = self.knn[idx, :40].astype(dtype=int)
                    for num,j in enumerate(neighbour):
                        if num == 0:
                            continue
                        self.nodes_edge.append([idx, j])

                    if idx % 10000 == 0:
                        print("calculate confidence (" + str(idx) + "/" + str(self.inst_num) + ")")
            else:
                for idx in range(self.inst_num):
                    first_nei_id = -1
                    # get node confidence
                    node_conf = self.conf[idx]
                    # get neighbour id
                    neighbour = self.knn[idx, :].astype(dtype=int)
                    # get neighbour confidence
                    nei_conf = self.conf[neighbour]
                    # find the first nearest neighbour
                    nei_idx = np.where(nei_conf > node_conf)
                    if len(nei_idx[0]):
                        first_nei_id = neighbour[nei_idx[0][0]]
                    else:
                        first_nei_id = -1
                    if first_nei_id != -1:
                        self.nodes_edge.append([idx, first_nei_id])
                    if idx % 10000 == 0:
                        print("calculate confidence (" + str(idx) + "/" + str(self.inst_num) + ")")

            if self.phase == 'train':
                gt = self.label[np.array(self.nodes_edge)[:, 0]] == self.label[np.array(self.nodes_edge)[:, 1]]
                print("total edge: ", len(gt))
                print("pos edge sum: ", gt.sum())
                print("neg edge sum: ", len(gt) - gt.sum())

                pos_idx = np.where(gt==True)[0]
                neg_idx = np.where(gt==False)[0]

                choose_pos = random.sample(list(np.arange(0, len(pos_idx))), 1000000)
                choose_neg = random.sample(list(np.arange(0, len(neg_idx))), 1000000)

                pos_edge = np.array(self.nodes_edge)[pos_idx[choose_pos]]
                neg_edge = np.array(self.nodes_edge)[neg_idx[choose_neg]]

                self.choose_nodes_edge = np.concatenate((pos_edge, neg_edge))
            else:
                gt = self.label[np.array(self.nodes_edge)[:, 0]] == self.label[np.array(self.nodes_edge)[:, 1]]
                print("total edge: ", len(gt))
                print("pos edge sum: ", gt.sum())
                print("neg edge sum: ", len(gt) - gt.sum())

                self.choose_nodes_edge = self.nodes_edge

    def __len__(self):
        return len(self.choose_nodes_edge)


    def __getitem__(self, index):
        edge = self.choose_nodes_edge[index]
        f_node_id = edge[0]
        s_node_id = edge[1]
        f_feature = torch.tensor(self.feature[f_node_id])
        s_feature = torch.tensor(self.feature[s_node_id])
        f_k_node_id = self.knn[f_node_id, :self.k]
        s_k_node_id = self.knn[s_node_id, :self.k]
        f_k_feature = torch.tensor(self.feature[f_k_node_id])
        s_k_feature = torch.tensor(self.feature[s_k_node_id])
        gt = int(self.label[f_node_id] == self.label[s_node_id])

        if self.phase == 'train':
            return f_feature, s_feature, f_k_feature, s_k_feature, gt
        else:
            return f_feature, s_feature, f_k_feature, s_k_feature, gt, f_node_id, s_node_id


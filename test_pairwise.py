import torch
import numpy as np
import torch.nn as nn

from utils import Timer, build_knn, knn2spmat
from Dataset import build_dataset
from Dataloader import build_dataloader
from evaluate import evaluate
from torch.utils.data import DataLoader


def _find_parent(parent, u):
    idx = []
    # parent is a fixed point
    while (u != parent[u]):
        idx.append(u)
        u = parent[u]
    for i in idx:
        parent[i] = u
    return u

def edge_to_connected_graph(edges, num):
    parent = list(range(num))
    for u, v in edges:
        v = int(v)
        p_u = _find_parent(parent, u)
        p_v = _find_parent(parent, v)
        parent[p_u] = p_v

    for i in range(num):
        parent[i] = _find_parent(parent, i)
    remap = {}
    uf = np.unique(np.array(parent))
    for i, f in enumerate(uf):
        remap[f] = i
    cluster_id = np.array([remap[f] for f in parent])
    return cluster_id

def add_edge(pred, bid, cid):
    edge = []
    for idx, (bid_, cid_) in enumerate(zip(bid, cid)):
        if pred[idx] == 1:
            edge.append([bid_, cid_])
    return edge


def test_pairwise(model, cfg):
    with Timer('1.prepare Dataset'):
        dataset = build_dataset(cfg.model['type'], cfg)

        data_loader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.workers, pin_memory=True,
                                 shuffle=True)

        criterion = nn.CrossEntropyLoss().cuda()

        model = model.cuda()

        model.load_state_dict(torch.load(cfg.save_path))
        print("=> Loaded checkpoint model '{}'".format(cfg.save_path))
        model.eval()

        edges = []

        with Timer('3.calculate edge:"'):
            # cal cb node
            gt_one = 0
            gt_zero = 0
            r_one = 0
            r_zero = 0

            for i, (b_feature, c_feature, b_k_feature, c_k_feature, gt, bid, cid) in enumerate(data_loader):
                b_feature, c_feature, b_k_feature, c_k_feature, gt = map(lambda x: x.cuda(), (
                b_feature, c_feature, b_k_feature, c_k_feature, gt))
                pred = model(b_feature, c_feature, b_k_feature, c_k_feature)
                loss = criterion(pred, gt)

                pred = torch.argmax(pred, dim=1).long()
                acc = torch.mean((pred == gt).float())

                for p, g in zip(pred, gt):
                    if g == 1:
                        gt_one += 1
                        if p == 1:
                            r_one += 1
                    else:
                        gt_zero += 1
                        if p == 0:
                            r_zero += 1

                print('loss:{:.8f},ACC:{:.4f},{}/{}'.format(loss.item(), acc, i,
                                                            int(dataset.inst_num / cfg.batch_size)))

                edge_new = add_edge(pred, bid, cid)

                edges = edges + edge_new

            print("model1 gt_one:", gt_one)
            print("model1 gt_zero:", gt_zero)
            print("model1 right:", r_one + r_zero)
            print("gt1 and 1:", r_one)
            print("gt0 and 0:", r_zero)
            print("gt1 but 0:", gt_one - r_one)
            print("gt0 but 1:", gt_zero - r_zero)

        edges = np.array(edges)
        pre_labels = edge_to_connected_graph(edges, dataset.inst_num)

        labels = torch.LongTensor(dataset.label)
        print("---------------------------------------------------")
        pre, rec, fscore = evaluate(labels, pre_labels, 'pairwise')
        print("P F-score: pre:", pre, " recall:", rec, "F-score:", fscore)
        print("---------------------------------------------------")
        pre, rec, fscore = evaluate(labels, pre_labels, 'bcubed')
        print("Bcubed F-score: pre:", pre, " recall:", rec, "F-score:", fscore)
        print("---------------------------------------------------")
        nmi = evaluate(labels, pre_labels, 'nmi')
        print("NMI:", nmi)
        print("---------------------------------------------------")
        print("nearest")

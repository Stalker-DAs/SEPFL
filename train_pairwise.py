import math

import torch
import numpy as np
import torch.optim as optim
import random
import os

import torch.nn as nn
from torch.utils.data import DataLoader
from utils import Timer, build_knn, knn2spmat
from Dataset import build_dataset


def schedule_lr(optimizer):
    for params in optimizer.param_groups:
        params['lr'] /= 10.


def calculate_ACC(pred, gt):
    N = len(gt)
    pred_label = torch.argmax(pred, dim=-1)
    return sum(pred_label == gt) / N


def train_pairwise(model, cfg):
    dataset = build_dataset(cfg.model['type'], cfg)

    data_loader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.workers, pin_memory=True, shuffle=True)

    OPTIMIZER = optim.SGD([{'params': model.parameters(), 'weight_decay': cfg.optimizer['weight_decay']}],
                          lr=cfg.optimizer['lr'], momentum=cfg.optimizer['momentum'])

    model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    with Timer('train'):
        for epoch in range(cfg.epochs):
            if epoch == cfg.STAGES[0]:
                schedule_lr(OPTIMIZER)
            if epoch == cfg.STAGES[1]:
                schedule_lr(OPTIMIZER)
            if epoch == cfg.STAGES[2]:
                schedule_lr(OPTIMIZER)

            count = 0
            print("epoch: ", epoch)

            gt_one = 0
            gt_zero = 0
            r_one = 0
            r_zero = 0

            for i, (bf, cf, bkf, ckf, gt) in enumerate(data_loader):
                bf, cf, bkf, ckf, gt= map(lambda x: x.cuda(), (bf, cf, bkf, ckf, gt))

                pred = model(bf, cf, bkf, ckf)
                loss = criterion(pred, gt)

                OPTIMIZER.zero_grad()
                loss.backward()
                OPTIMIZER.step()

                pred = torch.argmax(pred, dim=1).long()
                acc = torch.sum((pred == gt).float())
                acc = acc / cfg.batch_size

                if epoch % cfg.print_epoch == 0:
                    if cfg.print_results:
                        for p, g in zip(pred, gt):
                            if g == 1:
                                gt_one += 1
                                if p == 1:
                                    r_one += 1
                            else:
                                gt_zero += 1
                                if p == 0:
                                    r_zero += 1

                count += 1
                if i % cfg.print_iter == 0:
                    print('loss:{:.8f},ACC:{:.4f},{}/{}'.format(loss.item(), acc, i, int(2000000 / cfg.batch_size)))

            if epoch % cfg.print_epoch == 0:
                print("cb nodes:")
                print("total gt_one:", gt_one)
                print("total gt_zero:", gt_zero)
                print("gt1 and 1:", r_one)
                print("gt0 and 0:", r_zero)
                print("gt1 but 0:", gt_one - r_one)
                print("gt0 but 1:", gt_zero - r_zero)

            if epoch % cfg.print_epoch == 0:
                torch.save(model.state_dict(), os.path.join(cfg.save_path, "model_{}.pth".format(epoch)))
                print("save model epoch:", epoch)

        torch.save(model.state_dict(), os.path.join(cfg.save_path, "model_{}.pth".format(cfg.epochs)))
        print("save model epoch:", cfg.epochs)
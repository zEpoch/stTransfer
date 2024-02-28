#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/23/24 11:03 AM
# @Author  : zhoutao3
# @File    : focal_loss.py
# @Software: VSCode
# @Emial   : zhoutao3@genomics.cn
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiCEFocalLoss(nn.Module):
    def __init__(self, class_num, gamma=2, alpha=.25, reduction="mean"):
        super(MultiCEFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = 1.
        else:
            self.alpha = alpha

        self.gamma = gamma
        self.reduction = reduction
        self.class_num = class_num

    def forward(self, predict, target):
        pt = F.softmax(predict, dim=1)
        class_mask = F.one_hot(target, self.class_num)
        prob = (pt * class_mask).sum(1).view(-1, 1)
        log_p = prob.log()
        loss = -self.alpha * (torch.pow(1 - prob, self.gamma)) * log_p

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss



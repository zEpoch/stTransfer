#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/23/24 10:29 AM
# @Author  : zhoutao3
# @File    : cell_type_ann_model.py
# @Software: VSCode
# @Email   : zhoutao3@genomics.cn

import time
import warnings
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, VGAE, GATConv,GATv2Conv,GraphConv,GINConv, SAGEConv # type: ignore
from torch_geometric.utils import remove_self_loops, add_self_loops, negative_sampling # type: ignore
import xgboost as xgb # type: ignore
from xgboost import XGBClassifier # type: ignore
warnings.filterwarnings("ignore")

'''
PyTorch Geometric (PyG) 提供了许多图神经网络层，其中一些可以直接替换 GCNConv 而不需要更改参数。以下是一些例子：

ChebConv: ChebNet 卷积层，需要额外的参数 K，表示使用的切比雪夫多项式的阶数。

SAGEConv: GraphSAGE 卷积层。

GATConv: 图注意力网络 (GAT) 卷积层。发现很容易 聚合 临近节点的信息

GINConv: 图同构网络 (GIN) 卷积层。无法正确调用

GraphConv: 图卷积网络层，这是一个更一般的图卷积操作。发现 Psuedo-Acc 非常的低

ARMAConv: 自回归移动平均 (ARMA) 卷积层。

'''

class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphEncoder, self).__init__()
        # self.gc_feat = GCNConv(in_channels, hidden_channels)
        # self.gc_mean = GCNConv(hidden_channels, out_channels)
        # self.gc_logstd = GCNConv(hidden_channels, out_channels)
        # self.gc_feat = GATConv(in_channels, hidden_channels)
        # self.gc_mean = GATConv(hidden_channels, out_channels)
        # self.gc_logstd = GATConv(hidden_channels, out_channels)
        self.gc_feat = GCNConv(in_channels, hidden_channels,)
        self.gc_mean = GATConv(hidden_channels, out_channels)
        self.gc_logstd = GATConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, edge_weight):
        x = self.gc_feat(x, edge_index, edge_weight).relu()
        mean = self.gc_mean(x, edge_index, edge_weight)
        logstd = self.gc_logstd(x, edge_index, edge_weight)
        return mean, logstd
'''
from torch.nn import Sequential, Linear, ReLU

class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphEncoder, self).__init__()

        nn1 = Sequential(Linear(in_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.gc_feat = GINConv(nn1)
        # x = self.gc_feat(x, edge_index)

        nn2 = Sequential(Linear(hidden_channels, out_channels), ReLU(), Linear(out_channels, out_channels))
        self.gc_mean = GINConv(nn2)
        # mean = self.gc_mean(x, edge_index)

        nn3 = Sequential(Linear(hidden_channels, out_channels), ReLU(), Linear(out_channels, out_channels))
        self.gc_logstd = GINConv(nn3)
        # logstd = self.gc_logstd(x, edge_index)

    def forward(self, x, edge_index):
        x = self.gc_feat(x, edge_index).relu()
        mean = self.gc_mean(x, edge_index)
        logstd = self.gc_logstd(x, edge_index)
        return mean, logstd
'''

def full_block(in_features, out_features, drop_rate=0.2):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        nn.ELU(),
        nn.Dropout(p=drop_rate)
    )


class KDLoss(nn.Module):
    def __init__(self, T):
        super(KDLoss, self).__init__()
        self.T = T

    def forward(self, input, target):
        return nn.KLDivLoss()(
            (input / self.T).log_softmax(1),
            (target / self.T).softmax(1)
        ) * self.T * self.T

'''
class SpatialModel(nn.Module):
    def __init__(self, input_dim, num_classes, gae_dim, dae_dim, feat_dim):
        super(SpatialModel, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.gae_dim = gae_dim
        self.dae_dim = dae_dim
        self.feat_dim = feat_dim
        self.fcat_dim = self.dae_dim[1] + self.gae_dim[1]
        self.encoder = nn.Sequential(full_block(self.input_dim, self.dae_dim[0]),
                                     full_block(self.dae_dim[0], self.dae_dim[1]))
        self.decoder = nn.Linear(self.feat_dim, self.input_dim)
        self.vgae = VGAE(GraphEncoder(self.dae_dim[1], self.gae_dim[0], self.gae_dim[1]))
        self.feat_fc_x = nn.Sequential(nn.Linear(self.fcat_dim, self.feat_dim), nn.ELU())
        self.feat_fc_g = nn.Sequential(nn.Linear(self.fcat_dim, self.feat_dim), nn.ELU())
        self.classifier = nn.Linear(self.fcat_dim, self.num_classes)

    def forward(self, x, edge_index, edge_weight):
        feat_x = self.encoder(x)
        feat_g = self.vgae.encode(feat_x, edge_index, edge_weight)
        feat = torch.cat([feat_x, feat_g], 1)
        feat_x = self.feat_fc_x(feat)
        feat_g = self.feat_fc_g(feat)
        x_dec = self.decoder(feat_x)
        dae_loss = F.mse_loss(x_dec, x)
        gae_loss = self.recon_loss(feat_g, edge_weight, edge_index) + 1 / len(x) * self.vgae.kl_loss()
        cls = self.classifier(feat)
        return cls, dae_loss, gae_loss

    def recon_loss(self, z, edge_weight, pos_edge_index, neg_edge_index=None):
        pos_dec = self.vgae.decoder(z, pos_edge_index, sigmoid=False)
        pos_loss = F.binary_cross_entropy_with_logits(pos_dec, edge_weight)
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_dec = self.vgae.decoder(z, neg_edge_index, sigmoid=False)
        neg_loss = -F.logsigmoid(-neg_dec).mean()
        return pos_loss + neg_loss

'''
from torch_geometric.nn import GAE, GCNConv

class SpatialModel(nn.Module):
    def __init__(self, input_dim, num_classes, gae_dim, dae_dim, feat_dim):
        super(SpatialModel, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.gae_dim = gae_dim
        self.dae_dim = dae_dim
        self.feat_dim = feat_dim
        self.fcat_dim = self.dae_dim[1] + self.gae_dim[1]
        self.encoder = nn.Sequential(full_block(self.input_dim, self.dae_dim[0]),
                                     full_block(self.dae_dim[0], self.dae_dim[1]))
        self.decoder = nn.Linear(self.feat_dim, self.input_dim)
        self.gae = GAE(GCNConv(self.dae_dim[1], self.gae_dim[1]))  # Replace VGAE with GAE
        self.feat_fc_x = nn.Sequential(nn.Linear(self.fcat_dim, self.feat_dim), nn.ELU())
        self.feat_fc_g = nn.Sequential(nn.Linear(self.fcat_dim, self.feat_dim), nn.ELU())
        self.classifier = nn.Linear(self.fcat_dim, self.num_classes)

    def forward(self, x, edge_index, edge_weight):
        feat_x = self.encoder(x)
        feat_g = self.gae.encode(feat_x, edge_index)  # Use GAE to encode graph features
        feat = torch.cat([feat_x, feat_g], 1)
        feat_x = self.feat_fc_x(feat)
        feat_g = self.feat_fc_g(feat)
        x_dec = self.decoder(feat_x)
        dae_loss = F.mse_loss(x_dec, x)
        gae_loss = self.recon_loss(feat_g, edge_weight, edge_index)  # No KL divergence loss
        cls = self.classifier(feat)
        return cls, dae_loss, gae_loss

    def recon_loss(self, z, edge_weight, pos_edge_index, neg_edge_index=None):
        pos_dec = self.gae.decoder(z, pos_edge_index, sigmoid=False)
        pos_loss = F.binary_cross_entropy_with_logits(pos_dec, edge_weight)
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_dec = self.gae.decoder(z, neg_edge_index, sigmoid=False)
        neg_loss = -F.logsigmoid(-neg_dec).mean()
        return pos_loss + neg_loss


class SpatialModelTrainer:
    def __init__(self, input_dim, num_classes, device, KD_T):
        self.scaler = None
        self.scheduler = None
        self.optimizer = None
        self.criterion = None
        self.model = None
        self.device = device
        self.KD_T = KD_T
        self.set_model(input_dim, num_classes, KD_T)
        self.set_optimizer()

    def set_model(self, input_dim, num_classes, KD_T):
        gae_dim, dae_dim, feat_dim = [32, 8], [100, 20], 64
        self.model = SpatialModel(input_dim, num_classes, gae_dim, dae_dim, feat_dim).to(self.device)
        self.criterion = KDLoss(KD_T)

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=0.0001) # type: ignore
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=1.0)
        self.scaler = torch.cuda.amp.GradScaler()

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model']) # type: ignore
        self.optimizer.load_state_dict(checkpoint['optimizer']) # type: ignore
        print('  Load model from', path)

    def save_checkpoint(self, path):
        state = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()} # type: ignore
        torch.save(state, path)
        print('  Model is saved in', path)

    def train(self, data, epochs, w_cls, w_dae, w_gae):
        self.model.train() # type: ignore
        start_time = time.time()
        for epoch in range(1, epochs + 1):
            data = data.to(self.device, non_blocking=True)
            inputs, targets = data.x, data.y
            edge_index = data.edge_index
            edge_weight = data.edge_weight
            with torch.cuda.amp.autocast():
                outputs, dae_loss, gae_loss = self.model(inputs, edge_index, edge_weight) # type: ignore
                loss = w_cls * self.criterion(outputs, targets) + w_dae * dae_loss + w_gae * gae_loss # type: ignore
            train_loss = loss.item()
            self.optimizer.zero_grad() # type: ignore
            self.scaler.scale(loss).backward() # type: ignore
            self.scaler.step(self.optimizer) # type: ignore
            self.scaler.update() # type: ignore
            total = targets.size(0)
            predictions = outputs.argmax(1)
            correct = predictions.eq(targets.argmax(1)).sum().item()
            self.scheduler.step() # type: ignore
            process_time = time.time() - start_time
            accuracy = correct / total * 100.0
            print('  [Epoch %3d] Loss: %.5f, Time: %.2f s, Psuedo-Acc: %.2f%%'
                  % (epoch, train_loss, process_time, accuracy))

    def valid(self, data):
        self.model.eval() # type: ignore
        with torch.no_grad():
            data = data.to(self.device)
            inputs = data.x
            edge_index = data.edge_index
            edge_weight = data.edge_weight
            outputs, _, _ = self.model(inputs, edge_index, edge_weight) # type: ignore
            predictions = outputs.argmax(1)
        predictions = predictions.detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy()
        return predictions,outputs

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 25/2/24 21:54 PM
# @Author  : zhoutao3
# @File    : dnn_entry.py
# @Email   : zhoutao3@genomics.cn
import os
import os.path as osp
import scanpy as sc # type: ignore
import scipy.sparse as sp
import torch
import time
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from .cell_type_ann_model import DNNModel,DNNModelWithAttention # type: ignore
from .focal_loss import MultiCEFocalLoss # type: ignore


class DNNTrainer:
    def __init__(self, input_dims, num_classes, gpu):
        self.set_device(gpu)
        self.set_model(input_dims, hidden_dims=1024, output_dims=num_classes)
        self.set_optimizer()

    def set_model(self, input_dims, hidden_dims, output_dims):
        self.model = DNNModelWithAttention(input_dims, hidden_dims, output_dims).to(self.device)

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=5e-4)

    def set_device(self, gpu=None):
        if gpu is not None and torch.cuda.is_available():
            self.device = torch.device("cuda:{}".format(gpu))
        else:
            self.device = torch.device("cpu")

    def save_model(self, marker_genes, batch_size, label_names, path):
        state = {'model': self.model,
                 'optimizer': self.optimizer.state_dict(),
                 'marker_genes': marker_genes,
                 'batch_size': batch_size,
                 'label_names': label_names
                 }
        torch.save(state, path)
        print(f"  [{time.strftime('%Y-%m-%d %H:%M:%S')} Model is saved in: {path}]")

    def train(self, data_loader, marker_genes=None, class_nums=None, batch_size=4096, label_names=None, epochs=200, gamma=2, alpha=.25, path="dnn.bgi"):
        self.model.train()
        best_loss = np.inf
        for epoch in range(epochs):
            epoch_acc = []
            epoch_loss = []
            for idx, data in enumerate(data_loader):
                inputs, targets = data
                inputs = inputs.to(self.device)
                targets = targets.long().to(self.device)
                output = self.model(inputs)
                loss = MultiCEFocalLoss(class_num=class_nums, gamma=gamma, alpha=alpha, reduction="mean")(output, targets)
                train_loss = loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total = targets.size(0)
                prediction = output.argmax(1)
                correct = prediction.eq(targets).sum().item()

                accuracy = correct / total * 100.
                epoch_acc.append(accuracy)
                epoch_loss.append(train_loss)
            print(f"  [{time.strftime('%Y-%m-%d %H:%M:%S')} Epoch: {epoch+1:3d} Loss: {np.mean(epoch_loss):.5f}, acc: {np.mean(epoch_acc):.2f}%]")
            if np.mean(epoch_loss) < best_loss:
                best_loss = np.mean(epoch_loss)
                self.save_model(marker_genes, batch_size, label_names, path)

    def validation(self, data_loader, model_path):
        checkpoint = torch.load(model_path)
        label_names = checkpoint['label_names']
        dnn_model = checkpoint["model"].to(self.device)
        dnn_model.eval()

        dnn_predictions = []
        val_acc = []
        with torch.no_grad():
            for idx, data in enumerate(data_loader):
                inputs, targets = data
                inputs = inputs.to(self.device)
                targets = targets.long().to(self.device)
                outputs = dnn_model(inputs)
                dnn_predictions.append(outputs.detach().cpu().numpy())

                total = targets.size(0)
                prediction = outputs.argmax(1)
                correct = prediction.eq(targets).sum().item()
                accuracy = correct / total * 100.
                val_acc.append(accuracy)
                pseudo_class = pd.Categorical([label_names[i] for i in dnn_predictions[-1].argmax(1)])
                print(f"  [{time.strftime('%Y-%m-%d %H:%M:%S')} accuracy: {accuracy:.2f}% \npseudo_class: {pseudo_class}")
            print(f"  [{time.strftime('%Y-%m-%d %H:%M:%S')} total accuracy: {np.mean(val_acc):.2f}%")


class DNNDataset(Dataset):
    def __init__(self, adata, ann_key, marker_genes=None):
        self.adata = adata
        self.shape = adata.shape
        self.ann_key = ann_key
        if sp.issparse(adata.X):
            adata.X = adata.X.toarray()

        if marker_genes is None:
            data = adata.X
        else:
            gene_indices = adata.var_names.get_indexer(marker_genes)
            data = np.pad(adata.X, ((0, 0), (0, 1)))[:, gene_indices].copy()

        norm_factor = np.linalg.norm(data, axis=1, keepdims=True)
        norm_factor[norm_factor == 0] = 1
        self.data = data / norm_factor

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx].squeeze()
        y = self.adata.obs[self.ann_key].cat.codes[idx]
        return x, y


def transform_data_loader(adata, ann_key, marker_genes=None, batch_size=4096):
    dataset = DNNDataset(adata, ann_key, marker_genes=marker_genes)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=16)
    return train_loader


def dnn_workflow(data_path,
                 ann_key,
                 marker_genes=None,
                 batch_size=4096,
                 epochs=200,
                 gpu="0",
                 model_name="dnn.bgi",
                 model_path="./output",
                 filter_mt=False,
                 cell_min_counts=300,
                 gene_min_cells=10,
                 cell_max_counts=98.):
    """
    :param data_path: data path, which must be AnnData format.
    :param ann_key: the annotation key in .obs.keys() list.
    :param marker_genes: whether to use marker list data to train the model. If None, all data is used to train the model. Default, None.
    :param batch_size:
    :param epochs:
    :param gpu: whether to use GPU training model. If None, the CPU training model is used. If it is number, the corresponding GPU training model is invoked.
    :param model_name:
    :param model_path: save dnn model path.
    :param filter_mt: whether to filter MT- gene.
    :param cell_min_counts:
    :param gene_min_cells:
    :param cell_max_counts: filter cell counts outliers.  If the value is 100, no filtering is performed. Range: (0, 100).
    :return:
    """
    os.makedirs(model_path, exist_ok=True)
    assert data_path.endswith(".h5ad"), "Error, Got an invalid DATA_PATH!"
    adata = sc.read_h5ad(data_path)
    print(f"  [Data Info] \n {adata}")
    assert batch_size <= adata.shape[0], "Error, Batch size cannot be larger than the data set row."

    if filter_mt:
        adata.var["mt"] = adata.var_names.str.startswith(["MT-", "mt-", "Mt-"])
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
        adata = adata[adata.obs["pct_counts_mt"] < 10].copy()
    if cell_min_counts > 0:
        sc.pp.filter_cells(adata, min_counts=cell_min_counts)
    if gene_min_cells > 0:
        sc.pp.filter_genes(adata, min_cells=gene_min_cells)
    if cell_max_counts < 100:
        max_count = np.percentile(adata.obs["nCount_RNA"], cell_max_counts)
        sc.pp.filter_cells(adata, max_counts=max_count)

    print(f"  [After Preprocessing Data Info] \n {adata}")
    adata.obs[ann_key] = adata.obs[ann_key].astype('category')
    label_names = adata.obs[ann_key].cat.categories.tolist()
    class_nums = len(adata.obs[ann_key].cat.categories)
    if marker_genes is None:
        marker_list = adata.var_names.tolist()
    else:
        marker_list = marker_genes

    data_loader = transform_data_loader(adata, ann_key, marker_genes, batch_size)

    trainer = DNNTrainer(input_dims=adata.shape[1],
                         num_classes=len(adata.obs[ann_key].cat.categories),
                         gpu=gpu)

    trainer.train(data_loader, marker_genes=marker_list, class_nums=class_nums, batch_size=batch_size,
                  label_names=label_names, epochs=epochs, path=osp.join(model_path, model_name))
    trainer.validation(data_loader, osp.join(model_path, model_name))


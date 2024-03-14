#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 25/2/24 21:54 PM
# @Author  : zhoutao3
# @File    : dnn_entry.py
# @Email   : zhoutao3@genomics.cn
'''
import os.path as osp
import pandas as pd
import scanpy as sc # type: ignore
import numpy as np
import scipy.sparse as sp
import random
import torch
import torch_geometric # type: ignore
from matplotlib import pyplot as plt

from .cell_type_ann_model import SpatialModelTrainer # type: ignore

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def load_data(data_path, filter_mt=True, min_cells=10, min_counts=300, max_percent=98.0):
    """
    loading and processing dataset
    :param data_path: Note that, the input data must be raw, can not do any preprocessing!
    :param filter_mt: Whether to filter MT- genes. default, True
    :param min_cells: Whether to filter genes. default, 10
    :param min_counts: Whether to filter cells. default, 300
    :param max_percent: Whether to filter cells, Range: (0, 100). default, 98.0
    :return:
    """
    print("======> Loading data...")
    adata = sc.read_h5ad(data_path)
    print('  Original Data Info: %d cells × %d genes.' % (adata.shape[0], adata.shape[1]))

    if filter_mt:
        adata.var["mt"] = adata.var_names.str.startswith(("MT-", "Mt-", "mt-"))
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
        adata = adata[adata.obs["pct_counts_mt"] < 10].copy() # type: ignore
    if min_counts > 0:
        sc.pp.filter_cells(adata, min_counts=min_counts)
    if min_cells > 0:
        sc.pp.filter_genes(adata, min_cells=min_cells)
    if max_percent < 100:
        max_counts = np.percentile(adata.X.sum(1).reshape(-1).tolist()[0], max_percent) # type: ignore
        sc.pp.filter_cells(adata, max_counts=max_counts)

    # adata_X_sparse_backup = adata.X.copy()
    print('  After Preprocessing Data Info: %d cells × %d genes.' % (adata.shape[0], adata.shape[1]))
    if sp.issparse(adata.X):
        adata.X = adata.X.toarray() # type: ignore
    return adata


def transfer_from_sc_data(adata, model_choice, model_path, gpu="0"):
    """
    :param adata:
    :param dnn_path: Pre-trained DNN model save path
    :param gpu: gpu number
    :return:
    """
    print("========> Transfering from sc-dataset...")
    if gpu is not None and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(gpu))
    else:
        device = torch.device("cpu")
    if model_choice == "dnn":
        checkpoint = torch.load(model_path)
        dnn_model = checkpoint["model"].to(device)
        dnn_model.eval()

        marker_genes = checkpoint["marker_genes"]
        gene_indices = adata.var_names.get_indexer(marker_genes)
        adata_X = np.pad(adata.X, ((0, 0), (0, 1)))[:, gene_indices].copy()
        norm_factor = np.linalg.norm(adata_X, axis=1, keepdims=True)
        norm_factor[norm_factor == 0] = 1
        dnn_inputs = torch.Tensor(adata_X / norm_factor).split(checkpoint["batch_size"])
        # Inference with DNN model.
        dnn_predictions = []
        with torch.no_grad():
            for batch_idx, inputs in enumerate(dnn_inputs):
                inputs = inputs.to(device)
                outputs = dnn_model(inputs)
                dnn_predictions.append(outputs.detach().cpu().numpy())
        label_names = checkpoint['label_names']
        adata.obsm['psuedo_label'] = np.concatenate(dnn_predictions)
        adata.obs['psuedo_class'] = pd.Categorical([label_names[i] for i in adata.obsm['psuedo_label'].argmax(1)])
        adata.uns['psuedo_classes'] = label_names
        return adata
    elif model_choice == "xgboost":
        import pickle
        with open(model_path, 'rb') as f:
            model, dic, marker_genes = pickle.load(f)
        gene_indices = adata.var_names.get_indexer(marker_genes)
        adata_X = np.pad(adata.X, ((0, 0), (0, 1)))[:, gene_indices].copy()
        xgboost_predictions = model.predict_proba(adata_X)
        adata.obsm['psuedo_label'] = xgboost_predictions
        adata.obs['psuedo_class'] = pd.Categorical([dic[i] for i in adata.obsm['psuedo_label'].argmax(1)])
        adata.uns['psuedo_classes'] = dic
        return adata
    return adata


def distribution_fine_tune(adata, pca_dim=200, k_graph=30, edge_weight=True, epochs=200, w_cls=20, w_dae=1., w_gae=1.,
                           gpu="0", save_path="./output"):
    """
    :param adata:
    :param pca_dim: PCA dims, default=200
    :param k_graph: neighbors number, default=30
    :param edge_weight: Add edge weight to the graph model, default=True
    :param epochs: GCN training epochs, default=200
    :param w_cls: class num weight, default=20
    :param w_dae: dnn weight
    :param w_gae: gcn weight
    :param gpu: gpu number
    :param save_path: results save path
    :return:
    """
    if gpu is not None and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(gpu))
    else:
        device = torch.device("cpu")

    print("========> Model Training...")
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    sc.pp.log1p(adata)
    adata.X = (adata.X - adata.X.mean(0)) / adata.X.std(0)
    gene_mat = torch.Tensor(adata.X)
    if pca_dim:
        u, s, v = torch.pca_lowrank(gene_mat, pca_dim)
        gene_mat = torch.matmul(gene_mat, v)
    # u, s, v = torch.pca_lowrank(gene_mat, pca_dim)
    # gene_mat = torch.matmul(gene_mat, v)
    cell_coo = torch.Tensor(adata.obsm['spatial'])
    data = torch_geometric.data.Data(x=gene_mat, pos=cell_coo) # type: ignore
    data = torch_geometric.transforms.KNNGraph(k=k_graph, loop=True)(data) # type: ignore
    data.y = torch.Tensor(adata.obsm['psuedo_label'])

    # Make distances as edge weights.
    if edge_weight:
        data = torch_geometric.transforms.Distance()(data) # type: ignore
        data.edge_weight = 1 - data.edge_attr[:, 0]
    else:
        data.edge_weight = torch.ones(data.edge_index.size(1))

    # Train self-supervision model.
    input_dim = data.num_features
    num_classes = len(adata.uns['psuedo_classes'])
    trainer = SpatialModelTrainer(input_dim, num_classes, device=device)
    trainer.train(data, epochs, w_cls, w_dae, w_gae)
    trainer.save_checkpoint(osp.join(save_path, "model.bgi"))

    # Inference.
    print('\n==> Inferencing...')
    predictions = trainer.valid(data)
    celltype_pred = pd.Categorical([adata.uns['psuedo_classes'][i] for i in predictions])

    # Save results.
    result = pd.DataFrame({'cell': adata.obs_names.tolist(), 'celltype_pred': celltype_pred})
    result.to_csv(osp.join(save_path, "model.csv"), index=False)
    adata.obs['celltype_pred'] = pd.Categorical(celltype_pred) # type: ignore
    # adata.X = adata_X_sparse_backup

    # --------------------------------------------------
    adata.obsm["X_pca"] = gene_mat.detach().cpu().numpy()
    adata.uns = None
    adata.write(osp.join(save_path, "adata.h5ad"))

    # Save visualization.
    spot_size = 30
    psuedo_top100 = adata.obs['psuedo_class'].to_numpy()
    other_classes = list(pd.value_counts(adata.obs['psuedo_class'])[100:].index)
    psuedo_top100[adata.obs['psuedo_class'].isin(other_classes)] = 'Others'
    adata.obs['psuedo_top100'] = pd.Categorical(psuedo_top100)
    sc.pl.spatial(adata, img_key=None, color=['psuedo_top100'], spot_size=spot_size, show=False)
    plt.savefig(osp.join(save_path, "psuedo_top100.pdf"), bbox_inches='tight', dpi=150)
    sc.pl.spatial(adata, img_key=None, color=['celltype_pred'], spot_size=spot_size, show=False)
    plt.savefig(osp.join(save_path, "celltype_pred.pdf"), bbox_inches='tight', dpi=150)
    print("Done!")
'''
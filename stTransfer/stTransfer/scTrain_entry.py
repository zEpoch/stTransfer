#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 25/2/24 21:54 PM
# @Author  : zhoutao3
# @File    : *.py
# @Email   : zhotoa@foxmail.com


import scipy.sparse as sp
import numpy as np
import xgboost as xgb # type: ignore
import anndata as ad
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

from typing import Dict, Optional, Tuple, Sequence,List

def adata_precess(adata: ad.AnnData,):
    '''
    adata: ad.AnnData, scRNA-seq data
    '''
    data = adata.X.toarray() if sp.issparse(adata.X) else adata.X # Fix: Use .toarray() to convert sparse matrix to dense matrix
    
    norm_factor = np.linalg.norm(data, axis=1, keepdims=True) # Fix: Use np.linalg.norm to calculate the norm of each row
    norm_factor[norm_factor == 0] = 1
    data = data / norm_factor
    
    return data

def xgboost_train(X: np.ndarray, 
                  y: np.ndarray, 
                  save_path: str,
                  n_fold: int = 10,
                  gpu: Optional[str] = None,):
    '''
    X: np.ndarray, scRNA-seq data
    y: np.ndarray, cell type annotation
    save_path: str, model save path
    n_fold: int, number of folds for xgboost training
    '''
    if gpu is not None and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import KFold
    import pickle
    
    dic = {list(set(y))[i]:i for i in range(len(set(y)))}
    y = np.array([dic[i] for i in y])
    reverse_dic = {v:k for k,v in dic.items()}
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('########--- model init ---##########')
    model = xgb.XGBClassifier(objective='multi: softmax', n_estimators=100, seed=42, device=device )
    kf = KFold(n_splits=n_fold, random_state=42, shuffle=True)
    # 用于存储每折的分数
    scores = []
    print('########--- model train ---##########')
    count = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 在训练集上训练模型
        model.fit(X_train, y_train, verbose=2)

        # 在测试集上评估模型
        score = model.score(X_test, y_test)
        print(f"Fold {count+1}: {score}")
        
        scores.append(score)
        count += 1

    # 打印平均分数
    print(f"Average score: {np.mean(scores)}")
    
    with open(osp.join(save_path,'xgboost_model.pkl'), 'wb') as f:
        pickle.dump((model, reverse_dic), f)
    return reverse_dic
    
def xgboost_fit(X: np.ndarray,
                save_path: str,
                dic: Dict,
                model_name: str = 'xgboost_model.pkl',):
    import pickle
    if model_name:
        with open(osp.join(save_path, model_name), 'rb') as f:
            model, dic = pickle.load(f)
    else:
        with open(osp.join(save_path, 'xgboost_model.pkl'), 'rb') as f:
            model, dic = pickle.load(f)
    
    psuedo_label = model.predict_proba(X)
    psuedo_class = [dic[i] for i in psuedo_label.argmax(1)]
    return psuedo_label, psuedo_class

import os
 
def mkdir(path):
 
	folder = os.path.exists(path)
	if not folder:                   
		os.makedirs(path)            
		print("---  new folder...  ---")
 
	else:
		print("---  There is this folder!  ---")
		
def distribution_fine_tune(X: np.ndarray, 
                           cell_coo: np.ndarray,
                           psuedo_label: np.ndarray,
                           psuedo_classes: Dict,
                           pca_dim: int = 500, 
                           k_graph: int = 30, 
                           edge_weight: bool = True, 
                           epochs: int = 200, 
                           w_cls: int = 50, 
                           w_dae: int = 1, 
                           w_gae: int = 1,
                           KD_T: float = 1.0,
                           gpu: Optional[str] = None, 
                           save_path: str = "./output"):
    """
    :param adata:
    :param pca_dim: PCA dims, default=200
    :param k_graph: neighbors number, default=30
    :param edge_weight: Add edge weight to the graph model, default=True
    :param epochs: GCN training epochs, default=200
    :param w_cls: class num weight, default=20
    :param w_dae: dnn weight
    :param w_gae: gcn weight
    :param gpu: gpu number or None for cpu
    :param save_path: results save path
    :return:
    """
    if gpu is not None and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(gpu))
    else:
        device = torch.device("cpu")

    print("========> Model Training...")
    
    gene_mat = torch.Tensor(X)
    if pca_dim:
        u, s, v = torch.pca_lowrank(gene_mat, pca_dim)
        gene_mat = torch.matmul(gene_mat, v)
    # u, s, v = torch.pca_lowrank(gene_mat, pca_dim)
    # gene_mat = torch.matmul(gene_mat, v)
    cell_coo = torch.Tensor(cell_coo) # type: ignore
    data = torch_geometric.data.Data(x=gene_mat, pos=cell_coo) # type: ignore
    data = torch_geometric.transforms.KNNGraph(k=k_graph, loop=True)(data) # type: ignore
    data.y = torch.Tensor(psuedo_label)

    # Make distances as edge weights.
    if edge_weight:
        data = torch_geometric.transforms.Distance()(data) # type: ignore
        data.edge_weight = 1 - data.edge_attr[:, 0]
    else:
        data.edge_weight = torch.ones(data.edge_index.size(1))

    # Train self-supervision model.
    input_dim = data.num_features
    num_classes = len(psuedo_classes)
    trainer = SpatialModelTrainer(input_dim, num_classes, device=device, KD_T = KD_T)
    trainer.train(data, epochs, w_cls, w_dae, w_gae)
    trainer.save_checkpoint(osp.join(save_path, "graph_finetune.bgi"))

    # Inference.
    print('\n==> Inferencing...')
    predictions, outputs = trainer.valid(data)
    pd.DataFrame(outputs, columns=list(psuedo_classes.values())).to_csv(osp.join(save_path, 'outputs.csv'))
    pd.DataFrame([psuedo_classes[i] for i in predictions]).to_csv(osp.join(save_path, 'celltype_pred.csv'))
    # celltype_pred.to_csv(osp.join(save_path, 'celltype_pred.csv'))
    return None
'''
def sc_model_train_test(sc_adata: ad.AnnData,
                        st_adata: ad.AnnData,
                        sc_ann_key: str,
                        save_path: str,
                        marker_genes: Optional[List[str]] = None,
                        k_n_fold: int = 10,
                        st_adata_spatial_key: str = 'spatial',
                        finetune_epochs: int = 50,
                        finutune_pca_dim: int = 500,
                        finutune_w_cls: int = 20,
                        finutune_w_dae: int = 1,
                        finutune_w_gae: int = 1,
                        gpu: Optional[str] = None,):
    mkdir(save_path)
    print('########--- pre process ---##########')
    sc_adata_var_names = sc_adata.var_names.tolist()
    st_adata_var_names = st_adata.var_names.tolist()
    common_genes = list(set(sc_adata_var_names) & set(st_adata_var_names))
    assert len(common_genes) > 0, 'No common genes between scRNA-seq and spatial transcriptomics data'
    sc_adata = sc_adata[:, sc_adata.var_names.isin(common_genes)]
    st_adata = st_adata[:, st_adata.var_names.isin(common_genes)]
    # re-index gene order
    sc_adata = sc_adata[:, np.array(common_genes).tolist()]
    st_adata = st_adata[:, np.array(common_genes).tolist()]
    
    st_adata = gaussian_smooth_adaptively(adata = st_adata)
    
    sc_X = adata_precess(sc_adata)
    ST_X = adata_precess(st_adata)
    sc_y = np.array(sc_adata.obs[sc_ann_key].to_list())
    print('########--- start trian ---##########')
    reverse_dic = xgboost_train(sc_X, sc_y, save_path, n_fold = k_n_fold, gpu=None if gpu is None else gpu)  # Fix: Handle the case when gpu is None
    
    psuedo_label, psuedo_class = xgboost_fit(ST_X, dic=reverse_dic, save_path=save_path)  # Fix: Pass the correct arguments to xgboost_fit
    psuedo_labelpd = pd.DataFrame(psuedo_label)
    psuedo_labelpd.columns = list(reverse_dic.values())  # Fix: Convert reverse_dic.values() to a list
    psuedo_labelpd.to_csv(osp.join(save_path, 'psuedo_label.csv'), )
    pd.DataFrame(psuedo_class).to_csv(osp.join(save_path, 'psuedo_class.csv'))
    cell_coo = st_adata.obsm[st_adata_spatial_key]
    distribution_fine_tune(ST_X, 
                           cell_coo=cell_coo, 
                           psuedo_label=psuedo_label,  # Add the missing argument "psuedo_label"
                           psuedo_classes=reverse_dic,  # Add the missing argument "psuedo_classes"
                           save_path=save_path,
                           epochs=finetune_epochs,
                           pca_dim=finutune_pca_dim,
                           w_cls=finutune_w_cls,
                           w_dae = finutune_w_dae,
                           w_gae = finutune_w_gae,
                           gpu=None if gpu is None else gpu)  # Fix: Handle the case when gpu is None
'''

# stTransfer

[![python >= 3.8](https://img.shields.io/badge/python-3.8-brightgreen)](https://www.python.org/) 

### Installation      
```python
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install stTransfer
```

```python
import stTransfer as st
import anndata as ad
sc_adata = ad.read('/home/share/huadjyin/home/zhoutao3/zhoutao3/stTransfer_1/example/data/mouse_testis_sc/preprocess/preprocess.h5ad')
st_adata = ad.read('/home/share/huadjyin/home/zhoutao3/zhoutao3/stTransfer_1/example/data/Mouse_spermatogenesis/processed/Diabetes_Slide-seq_data/Diabetes_1.h5ad')
sc_ann_key = 'celltype'
save_path = '/home/share/huadjyin/home/zhoutao3/zhoutao3/stTransfer_1/example/test/Mouse_spermatogenesis/Diabetes_1'

marker_genes = None

# st.sc_model_train_test(sc_adata, st_adata, sc_ann_key, save_path, marker_genes, finetune_epochs = 200,finutune_pca_dim = 0, gpu = 0) # if cuda
st.sc_model_train_test(sc_adata, st_adata, sc_ann_key, save_path, marker_genes, finetune_epochs = 200,finutune_pca_dim = 0 )
```
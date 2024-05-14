# stTransfer

[![python >= 3.8](https://img.shields.io/badge/python-3.8-brightgreen)](https://www.python.org/) 

### Installation      
```bash
conda create -n zt_stTranfer_test python=3.8
conda activate zt_stTranfer_test
conda install anaconda::h5py
pip install scvi-tools==0.13.0
pip install git+https://github.com/zEpoch/stTransfer
pip install pytorch_lightning==1.6.5
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
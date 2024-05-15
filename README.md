# stTransfer

[![python >= 3.8](https://img.shields.io/badge/python-3.8-brightgreen)](https://www.python.org/) 

### Installation      
```bash
conda create -n zt_stTranfer_test python=3.8
conda activate zt_stTranfer_test
conda install anaconda::h5py
pip install pytorch-lightning==1.6.5
pip install torch==1.13.1+cu117
pip install torch-cluster==1.6.1+pt113cu117
pip install git+https://github.com/zEpoch/stTransfer
pip install pytorch_lightning==1.6.5

pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-2.1.1+cpu.html
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.1.1+cpu.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-0.6.17+cpu.html
pip install torch-geometric==2.3.1
pip install torchmetrics==1.2.0


    "pytorch-lightning==1.6.5",
    "torch==1.13.1+cu117",
    "torch-cluster==1.6.1+pt113cu117",
    "torch-geometric==2.3.1",
    "torch-scatter==2.1.1+pt113cu117",
    "torch-sparse==0.6.17+pt113cu117",
    "torch-spline-conv==1.2.2+pt113cu117",
    "torchaudio==0.13.1+cu117",
    "torchmetrics==1.2.0",
    "torchvision==0.14.1+cu117",
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
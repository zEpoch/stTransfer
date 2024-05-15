# stTransfer

[![python >= 3.8](https://img.shields.io/badge/python-3.8-brightgreen)](https://www.python.org/) 

### Installation      
```bash
conda create -n stTranfer python=3.8
conda activate stTranfer
conda activate zt_stTransfer_2
conda install anaconda::h5py # if h5py install error
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# install for cuda manylinux2014_aarch64
pip install torch-cluster==1.6.1 -f https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu117.html
pip install torch-scatter==2.1.1 -f https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu117.html
pip install torch-sparse==0.6.17 -f https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu117.html
pip install torch-spline-conv==1.2.2 -f https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu117.html
pip install pyg-lib==0.2.0 -f https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu117.html
pip install torch-geometric

# install for cuda linux_x86_64
pip install torch-cluster==1.6.1+pt113cu117 -f https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu117.html
pip install torch-scatter==2.1.1+pt113cu117 -f https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu117.html
pip install torch-sparse==0.6.17+pt113cu117 -f https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu117.html
pip install torch-spline-conv==1.2.2+pt113cu117 -f https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu117.html
pip install pyg-lib==0.2.0+pt113cu117 -f https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu117.html
pip install torch-geometric

# if jaxlib error if you need
conda install jaxlib

pip install git+https://github.com/zEpoch/stTransfer
# need to install again
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# fix error you may need
pip install flax==0.7.2 
pip install pyro-ppl==1.8.6
pip install optax==0.1.7
```

And for my mechine, I need to install the following packages:
```bash
pip install wheel
git config --global url."https://mirror.ghproxy.com/https://github.com".insteadOf "https://github.com" 
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
pip install torch-cluster==1.6.1+pt113cu117 -f https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu117.html
pip install torch-scatter==2.1.1+pt113cu117 -f https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu117.html
pip install torch-sparse==0.6.17+pt113cu117 -f https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu117.html
pip install torch-spline-conv==1.2.2+pt113cu117 -f https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu117.html
pip install wheel
pip install pyro-ppl==1.8.6
git config --global url."https://mirror.ghproxy.com/https://github.com".insteadOf "https://github.com" 
pip install git+https://github.com/zEpoch/stTransfer
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install flax==0.7.2 
pip install pyro-ppl==1.8.6
pip install optax==0.1.7
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
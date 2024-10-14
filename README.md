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
pip install torch-cluster==1.6.1+pt113cu117 -f https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu117.html
pip install torch-scatter==2.1.1+pt113cu117 -f https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu117.html
pip install torch-sparse==0.6.17+pt113cu117 -f https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu117.html
pip install torch-spline-conv==1.2.2+pt113cu117 -f https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu117.html
pip install torch-geometric
git config --global url."https://mirror.ghproxy.com/https://github.com".insteadOf "https://github.com"
pip install git+https://github.com/zEpoch/stTransfer
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install flax==0.7.2 
pip install pyro-ppl==1.8.6
pip install optax==0.1.7
```

```python
from stTransfer import transfer

kwargs = {
    'sc_adata_path': '/data/work/sttransfer/stereoseq/script/adata.scvi.leiden.anno.h5ad',
    'sp_adata_path': '/data/work/sttransfer/cellbin_adatas/A01890F2_ot_left.h5ad',
    'sc_anno': 'anno',
    'sp_anno': 'annotation',
    'name': 'A01890F2_ot_left',
    'save_path': '/data/work/sttransfer/stereoseq/Result_3/',
    'sp_filter': True,
    'k_n_fold': 3, # for xgboost train
    'st_spatial_anno': 'spatial', # in the obsm
    'finetune_epochs': 50,
    'finutune_pca_dim': 0,
    'gpu': 0,
    'finutune_w_cls': 10,
    'finutune_w_gae': 1,
    'finutune_w_dae': 1,
'KD_T': 1,
'marker_genes':None
}
transfer(**kwargs)

import anndata as ad
adata = ad.read('/data/work/sttransfer/stereoseq/Result_3/spd_filtered.h5ad')
import pandas as pd
csv = pd.read_csv('/data/work/sttransfer/stereoseq/Result_3/celltype_label.h5ad')
csv.lolumns = ['0']
adata.obs['celltype'] = csv['0']
```
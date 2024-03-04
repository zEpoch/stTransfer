# stTransfer

[![python >= 3.8](https://img.shields.io/badge/python-3.8-brightgreen)](https://www.python.org/) 

### Installation      
```python
pip install stTransfer
```

```python
import stTransfer as st
st.dnn_workflow(data_path = '/data/input/single.h5ad',
                ann_key = 'celltype', # celltype in adata.obs
                marker_genes=None, # marker genes list
                batch_size=4096, # train batch size
                epochs=200, # train epochs
                gpu="0", # gpu id
                model_name="dnn.bgi", # model name
                model_path="/data/model", # model path
                filter_mt=False, # filter mitochondrial genes or not
                cell_min_counts=300, # min counts per cell
                gene_min_cells=10, # min cells per gene
                cell_max_counts=98.) # max counts per cell

st_adata = st.load_data(data_path = '/data/input/st_adata.h5ad', # obsm.['spatial'] is required
                        filter_mt=True, 
                        min_cells=10, 
                        min_counts=300, 
                        max_percent=98.0) # load data

st_adata_with_pslabel = st.transfer_from_sc_data(adata = st_adata, # adata with obsm.['spatial']
                                                 dnn_path = '/data/model/dnn.bgi', # dnn model path
                                                 gpu="0")

st.distribution_fine_tune(st_adata_with_pslabel, 
                          pca_dim=0, 
                          k_graph=30, 
                          edge_weight=True, 
                          epochs=100, 
                          w_cls=50, 
                          w_dae=1., 
                          w_gae=1.,
                          gpu="0", 
                          save_path="/data/output") # output path
```
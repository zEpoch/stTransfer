#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/28 16:09
# @Author  : zhoutao
# @File    : setup.py.py
# @Software: VScode
# @Email   : zhotoa@foxmail.com

import setuptools
from wheel.bdist_wheel import bdist_wheel

__version__ = "1.0.7"


class BDistWheel(bdist_wheel):
    def get_tag(self):
        return (self.python_tag, "none", "any")


cmdclass = {
    "bdist_wheel": BDistWheel,
}

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# requirements = open("requirements.txt").readline()
install_requires = [
    "anndata>=0.8.0",
    "matplotlib>=3.5.1",
    "matplotlib-inline>=0.1.3",
    "networkx>=2.7.1",
    "numpy>=1.21.5",
    "pandas>=1.4.2",
    "scanpy>=1.9.1",
    "scikit-learn>=1.0.2",
    "scipy>=1.8.0",
    "seaborn>=0.11.2",
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
    "scvi-tools==0.17.0",
    "xgboost>=2.0.0",
]
setuptools.setup(
    name = "stTransfer",
    version = "1.0.24",
    description = "Transfer learning for spatial transcriptomics data and single-cell RNA-seq data.",
    author = "zhoutao",
    author_email = "zhotoa@foxmail.com",
    url = "https://github.com/zepoch/stTransfer.git",
    python_requires=">=3.8",
    packages=setuptools.find_packages(),
    # install_requires=requirements,
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    cmdclass=cmdclass,
)
# Project structure
Python modules:
* `LocalDNAS.py`: impementation of both the SuperNet method and cost-aware regularizers
* `alternatives.py`: alternative modules considered in supermodules
* `utils.py`: utility module

Jupyter notebooks:
* `SuperNet_MBNV3.ipynb`: training strategy for MBNV3 on ImageNet-100 and application of SuperNet to MBNV3 targeting SE modules
* `SuperNet_regularizers.ipynb`: cost-aware optimizations of SuperNet on MBNV3 by considering cost-aware regularizers on number of parameters and FLOPs
* `model_comparison.ipynb`: comparison between standard MBNV3 and models achieved with the application of SuperNet 

# Run experiments
## 1 - Create conda enviroment
1. Create and activate the conda environment:
```console
conda create -n cv-project python==3.11 -y
conda activate cv-project
```
2. Install `requirements.txt` file:
```console
pip install -r requirements.txt
```
## 2 - Download ImageNet100 dataset in the project folder
1. Move to the project folder and download the dataset using Kaggle:
```console
kaggle datasets download ambityga/imagenet100
```
2. Unzip `imagenet100.zip`:
```console
mkdir ImageNet
unzip imagenet100.zip -d ImageNet
```

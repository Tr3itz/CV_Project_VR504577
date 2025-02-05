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

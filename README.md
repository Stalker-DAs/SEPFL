# Structure-enhanced pairwise feature learning for face clustering
## Introduction

This model presents a novel face clustering framework named SEPFL that performs data grouping at the pair level. Compared to graph-based approaches, our framework incorporates pairwise feature learning for connectivity classification, reducing the computational cost and alleviating the dependence on thresholds in the inference phase. As the experimental results, SEPFL is more competitive than other advanced methods and demonstrates effectiveness in other clustering tasks, such as fashion clustering.

The main framework of SEPFL is shown in the following:

<img src=image/fig.png width=1000 height=345 />

## Main Results
<img src=image/results.png width=900 height=355 />

## Requirements
* Python=3.6.8
* Pytorch=1.7.1
* Cuda=11.0

## Hardware
The hardware we used in this work is as follows:
* NVIDIA GeForce RTX 3090
* Intel Xeon Gold 6226R CPU@2.90GHz

## Datasets
Create a new folder for dataset:
```
cd SEPFL
mkdir data
```
After that, follow the link below to download the dataset and construct the data directory as follows:
```
|——data
   |——features
      |——part0_train.bin
      |——part1_test.bin
      |——...
      |——part9_test.bin
   |——labels
      |——part0_train.meta
      |——part1_test.meta
      |——...
      |——part9_test.meta
   |——knns
      |——part0_train/faiss_k_80.npz
      |——part1_test/faiss_k_80.npz
      |——...
      |——part9_test/faiss_k_80.npz
```
The MS1M and DeepFashion dataset at https://github.com/yl-1993/learn-to-cluster/blob/master/DATASET.md#supported-datasets.

## Training
You can use the following command to train the model directly. Alternatively, you can find the model configuration file in `./config/cfg_train.py`.
```
cd SEPFL
python main.py
```

## Testing
If you want to test the model, please set the config file to `cfg_test.py` in `main.py` and make sure the weight file is in the `./saves` folder. The pre-trained weight is in the path `./saves/checkpoint.pth`. After that, you can get the test results with the following command:
```
cd SEPFL
python main.py
```

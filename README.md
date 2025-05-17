# DFRNet using Deep Learning

##  Table of Contents

---
1. [Overview](#1-overview)  
2. [Installation](#2-installation)  
3. [Attention Fix for MXNet (Optional)](#3-attention-fix-for-mxnet-optional)  
4. [Dataset Preparation](#4-dataset-preparation)  
5. [Training model](#5-training-model)  
   - [5.1 Train model at ElasticFace](#51-train-model-at-elasticface)  
   - [5.2 Train model at AdaFace + PartialFC](#52-train-model-at-adaface--partialfc)  
6. [Result](#6-result)  
   - [6.1 ElasticFace](#61-elasticface)  
   - [6.2 AdaFace + PartialFC](#62-adaface--partialfc)  

---

## 1. Overview

ElasticArcFace is an advanced face recognition framework that integrates:

- **ArcFace (Additive Angular Margin Loss)**
- **ElasticNet Regularization (ElasticFace)**
- **AdaFace (Quality Adaptive Margin)**
- **Self-Attention modules** (for enhanced feature extraction)

The goal is to improve recognition accuracy, especially in difficult conditions such as pose variations, occlusions, or low-quality images.

---

## 2. Installation

Install the required dependencies:

```bash
!pip install tensorboard
!pip install easydict
!pip install mxnet
!pip install onnx
!pip install scikit-learn
```

## 3. Attention Fix for MXNet (Optional)
For some versions of MXNet, you might encounter compatibility warnings regarding bool.
Fix it using the following command:

```bash
sed -i 's/bool = onp.bool/bool = bool/' /usr/local/lib/python3.11/dist-packages/mxnet/numpy/utils.py

```

if you want to begin train at ElasticFace, there is the line called 
```bash
# if args.resume:
    #     try:
    #         backbone_pth = os.path.join(cfg.output, str(cfg.global_step) + "backbone.pth")
    #         backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(local_rank)))

    #         if rank == 0:
    #             logging.info("backbone resume loaded successfully!")
    #     except (FileNotFoundError, KeyError, IndexError, RuntimeError):
    #         logging.info("load backbone resume init, failed!")
```

You can uncomment it, start to train, i want to comment it to prevent the backbone after i trained cannot load it successfully after i use pretrained model


## 4. Dataset Preparation

ElasticArcFace is compatible with various face recognition datasets such as:

- **MS1MV2**

You can use the public datasets provided by [InsightFace Datasets Repository](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_).

### Example Datasets Link:
- 📦 MS1MV2  (List & Download Scripts):  
  https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_/

## 5. Training model
### 5.1. Train model at ElasticFace
#### a) Train at Kaggle with 2 GPU
```bash
!CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train.py 
```

#### b) Train at Kaggle with 1 GPU (optional device)
```bash
!CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train.py 
```

#### c) If you want to continue train (optional)
First, customize the line name called "global_step" to actual your model after have this
For example: 22744backbone.pth -> global_step = 22744
```bash
!CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train.py --resume 1
```
### 5.2. Train model at AdaFace + PartialFC
```bash
!CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nproc_per_node=2 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="127.0.0.1" \
  --master_port=1234 \
  train.py configs/ms1mv2_r50.py
```

## 6. Result
### 6.1. ElasticFace
## Results

## Training Results

| Global Step | Margin (M) | Epoch | Dataset   | LFW Accuracy (%) |
|-------------|------------|-------|-----------|------------------|
| 352532      | 0.5        | 30    | MS1MV2    | 99.117           |
| 358218      | 0.6        | 31    | MS1MV2    | 99.223           |
| 375276      | 0.6        | 32    | MS1MV2    | 99.167           |



### 6.2. AdaFace + PartialFC
## Results

## Training Results

| Global Step |Epoch | Dataset   | LFW Accuracy (%) |
|-------------|------|---------- |------------------|
| 11300       | 0    | MS1MV2    | 96.467           |
| 22700       | 1    | MS1MV2    | 97.783           |
| 34100       | 2    | MS1MV2    | 98.100           |
| 41000       | 3    | MS1MV2    | 98.100           |

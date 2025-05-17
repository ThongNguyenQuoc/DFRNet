# DFRNet: Deep Face Recognition Network for Large-Scale Identity Recognition

This is the **official implementation** of **DFRNet**, a scalable and robust deep learning framework for large-scale face recognition, built upon state-of-the-art methods including **PartialFC**, **AdaFace**, and **ElasticFace**.

---

## 🔥 Related Works and Foundations

Our approach leverages and integrates the strengths of several key methods:

- [Killing Two Birds With One Stone: Efficient and Robust Training of Face Recognition CNNs by Partial FC (CVPR-2022)](https://arxiv.org/abs/2203.15565)
- [AdaFace: Quality Adaptive Margin for Face Recognition (CVPR-2022)](https://arxiv.org/abs/2204.00964)
- [ElasticFace: Elastic Margin Loss for Deep Face Recognition (arXiv)](https://arxiv.org/pdf/2109.09416.pdf)

We modified and extended **PartialFC** to support both **AdaFace** and **ElasticFace** under both **single-machine** and **distributed training environments** for efficient large-scale training.

---

## Setup
This line code of 2 folder can be run at Kaggle or Python 3.11 

Install all required package:

```bash
!pip install -r requirement.txt
```

Clone at repo:
```bash
https://github.com/ThongNguyenQuoc/DFRNet.git
cd DFRNet/DFRNet
```

##  Data Preparation
We train model 'DFRNet' using:

### CASIA-Webface (10K ids/0.5M images) 
[GDrive](https://drive.google.com/file/d/1KxNCrXzln0lal3N4JiYl9cFOIhT78y1l/view?usp=sharing)

### MS1M-ArcFace (85K ids/5.8M images) (MS1MV2)

[GDrive](https://drive.google.com/file/d/1SXS4-Am3bsKSK615qbYdbA_FMVh3sAvR/view?usp=sharing)

### MS1M-RetinaFace (MS1MV3)

[GDrive](https://drive.google.com/file/d/1JgmzL9OLTqDAZE86pBgETtSQL4USKTFy/view?usp=sharing)


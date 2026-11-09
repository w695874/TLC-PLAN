

# TLC-Plan: A Two-Level Codebook Based Network for End-to-End Vector Floorplan Generation (Eurographics 2026)

[![arXiv](https://img.shields.io/badge/📃-arXiv%20-red.svg)](https://arxiv.org/abs/2602.07100)
[![webpage](https://img.shields.io/badge/🌐-Website%20-blue.svg)](https://onlinelibrary.wiley.com/doi/10.1111/cgf.70360) 


Biao Xiong, Zhen Peng, Ping Wang, Qiegen Liu, Xian Zhong

![image-20260105125726246](assets/image-20260105125726246.png)

> Automated floorplan generation aims to improve design quality, architectural efficiency, and sustainability by jointly modeling global spatial organization and precise geometric detail. However, existing approaches operate in raster space and rely on post hoc vectorization, which introduces structural inconsistencies and hinders end-to-end learning. Motivated by compositional spatial reasoning, we propose TLC-Plan, a hierarchical generative model that directly synthesizes vector floorplans from input boundaries, aligning with human architectural workflows based on modular and reusable patterns. TLC-Plan employs a two-level VQ-VAE to encode global layouts as semantically labeled room bounding boxes and to refine local geometries using polygon-level codes. This hierarchy is unified in a CodeTree representation, while an autoregressive transformer samples codes conditioned on the boundary to generate diverse and topologically valid designs, without requiring explicit room topology or dimensional priors. Extensive experiments show state-of-the-art performance on RPLAN dataset (FID = 1.84, MSE = 2.06) and leading results on LIFULL dataset. The proposed framework advances constraint-aware and scalable vector floorplan generation for real-world architectural applications. 


## Requirements

### Environment
- Linux
- Python 3.8
- CUDA >= 11.4
- GPU with 24 GB ram recommended

### Dependencies
- PyTorch >= 1.10
- Install pythonocc following the instruction [here](https://github.com/tpaviot/pythonocc-core) (use mamba if conda is too slow).
- Install other dependencies with ```pip install -r requirements.txt```

We also provide the [docker image](https://hub.docker.com/r/samxuxiang/skexgen). Note: only tested on CUDA 11.4. 

# 基于双层构架实现内墙预测
Loop层学习当个层之间的墙的关系，Profile层学习房间之间的关系

## codebook训练

分别训练profile和loop的codebook。读取data/loop/train.pkl数据，config.py中对应模型的参数设置。

```
python codebook/train.py --output proj_log/profile --batchsize 256 --format profile --device 0
python codebook/train.py --output proj_log/loop --batchsize 256 --format loop --device 0
```

根据训练好的模型提取出profile.pkl和loop.pkl

```
python codebook/extract_code.py --checkpoint proj_log/profile --format profile --epoch 250 --device 0
python codebook/extract_code.py --checkpoint proj_log/loop --format loop --epoch 250 --device 0
```

## 预测模型训练

这部分代码在文件夹gen下面，主要函数是cond_train.py和cond_generation.分别进行训练和预测。数据处理在dataset.py中CADData类。config.py中COND_TRAIN_EPOCH设置训练epoch和其他参数。

```
python gen/cond_train.py --output proj_log/gen_full --batchsize 256 --profile_code  profile.pkl --loop_code loop.pkl --mode cond --device 0
```

## 模型生成
读取训练好模型预测，

```
python gen/cond_generation.py --output result/ac_test --weight proj_log/gen_full --profile_code  profile.pkl --loop_code loop.pkl --mode cond --device 0
```






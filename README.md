<<<<<<< HEAD
# UFPMP-Det: Toward Accurate and Efficient Object Detection on Drone Imagery

The repo is the official implementation of  UFPMP-Det.

The code of **UFP** **module** is at [mmdet/core/ufp](mmdet/core/ufp)

The code of **MP-Det is** at [mmdet/models/dense_heads/mp_head.py](mmdet/models/dense_heads/mp_head.py)

The **config** of our project is at [configs/UFPMP-Det](configs/UFPMP-Det)

# Install

1. This repo is implemented based on [mmdetection](https://github.com/open-mmlab/mmdetection). Please install it according to [get_start.md](docs/en/get_started.md).
2. ```shell
   pip install nltk
   pip install albumentations
   ```
## Quickstart
We provide the Dataset(COCO Format) as follows:
- VisDrone:链接：https://pan.baidu.com/s/1FfAsAApHZruucO5A2QgQAg 提取码：qrvs
- UAVDT:链接：链接：https://pan.baidu.com/s/1KLmU5BBWwgtFbuZa7QWavw 提取码：z08x

We provide the checkpoint as follows:
- VisDrone Coarse-Det:链接: https://pan.baidu.com/s/1jK3bqImDGSwqRJGVXinS0w 提取码: nab3
- VisDrone MP-Det ResNet50: 链接: https://pan.baidu.com/s/1zOoJVO2fPejnzM9KioZLuQ 提取码: m7rj

# Training

This repo is only supposed single GPU.

## Prepare

Build by yourself: We provide two data set conversion tools.

```shell
# conver VisDrone to COCO
python UFPMP-Det-Tools/build_dataset/VisDrone2COCO.py
# conver UAVDT to COCO
python UFPMP-Det-Tools/build_dataset/UAVDT2COCO.py
# build UFP dataset(VisDrone)
CUDA_VISIBLE_DEVICES=2 python UFPMP-Det-Tools/build_dataset/UFP_VisDrone2COCO.py \
    ./configs/UFPMP-Det/coarse_det.py \
    ./work_dirs/coarse_det/epoch_12.pth \
    xxxxxx/dataset/COCO/images/UAVtrain \
    xxxxxx/dataset/COCO/annotations/instances_UAVtrain_v1.json \
    xxxxxx/dataset/COCO/images/instance_UFP_UAVtrain/ \
    xxxxxx/dataset/COCO/annotations/instance_UFP_UAVtrain.json \
    --txt_path path_to_VisDrone_annotation_dir
```

Download:

In Quick Start

## Train Coarse Detector

```shell
CUDA_VISIBLE_DEVICES=0 python tools/train.py ./configs/UFPMP-Det/coarse_det.py
```

## Train MP-Det

```shell
CUDA_VISIBLE_DEVICES=0 python tools/train.py ./config/UFPMP-Det/mp_det_res50.py
```

# Test

```shell
CUDA_VISIBLE_DEVICES=2 python UFPMP-Det-Tools/eval_script/ufpmp_det_eval.py \
    ./configs/UFPMP-Det/coarse_det.py \
    ./work_dirs/coarse_det/epoch_12.pth \
    ./configs/UFPMP-Det/mp_det_res50.py  \
    ./work_dirs/mp_det_res50/epoch_12.pth \
    XXXXX/dataset/COCO/annotations/instances_UAVval_v1.json \
    XXXXX/dataset/COCO/images/UAVval

```
## Citation

If you find our paper or this project helps your research, please kindly consider citing our paper in your publication.

```
@inproceedings{ufpmpdet,
  title={UFPMP-Det: Toward Accurate and Efficient Object Detection on Drone Imagery},
  author={Huang, Yecheng and Chen, Jiaxin and Huang, Di},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2022}
}
```

# TinyNAS
                                                                  
- This repository is a collection of training-free neural architecture search methods developed by TinyML team, Data Analytics and Intelligence Lab, Alibaba DAMO Academy. Researchers and developers can use this toolbox to design their neural architectures with different budgets on CPU devices within 30 minutes.
    - Training-Free Neural Architecture Evaluation Scores by Entropy [**DeepMAD**](https://arxiv.org/abs/2303.02165)(CVPR'23), and by Gradient [**Zen-NAS**](https://openaccess.thecvf.com/content/ICCV2021/papers/Lin_Zen-NAS_A_Zero-Shot_NAS_for_High-Performance_Image_Recognition_ICCV_2021_paper.pdf)(ICCV'21)
    - Joint Quantization and Architecture Search [**Mixed-Precision Quantization Search**](https://openreview.net/pdf?id=lj1Eb1OPeNw)(NeurIPS'22)
    - Application : Object Detection [**MAE-DET**](https://proceedings.mlr.press/v162/sun22c/sun22c.pdf)(ICML'22)
    - Application : Action Recognition [**Maximizing Spatio-Temporal Entropy**](https://openreview.net/pdf?id=lj1Eb1OPeNw)(ICLR'23)

## News

- **:sunny: Hiring research interns for Neural Architecture Search, Tiny Machine Learning, Computer Vision tasks: [xiuyu.sxy@alibaba-inc.com](xiuyu.sxy@alibaba-inc.com)**
- :boom: 2023.04: [**PreNAS: Preferred One-Shot Learning Towards Efficient Neural Architecture Search**](https://arxiv.org/abs/2304.14636) is accepted by ICML'23.
- :boom: 2023.04: We will give a talk on Zero-Cost NAS at [**IFML Workshop**](https://www.ifml.institute/events/ifml-workshop-2023), April 20, 2023.
- :boom: 2023.03: Code for [**E3D**](configs/action_recognition/README.md) is now released.
- :boom: 2023.03: The code is refactoried and DeepMAD is supported.
- :boom: 2023.03: [**DeepMAD: Mathematical Architecture Design for Deep Convolutional Neural Network**](https://arxiv.org/abs/2303.02165) is accepted by CVPR'23.
- :boom: 2023.02: A demo is available on [**ModelScope**](https://modelscope.cn/studios/damo/TinyNAS/summary)
- :boom: 2023.01: [**Maximizing Spatio-Temporal Entropy of Deep 3D CNNs for Efficient Video Recognition**](https://openreview.net/pdf?id=lj1Eb1OPeNw) is accepted by ICLR'23.
- :boom: 2022.11: [**DAMO-YOLO**](https://github.com/tinyvision/DAMO-YOLO) backbone search is now supported! And paper is on [ArXiv](https://arxiv.org/abs/2211.15444) now.
- :boom: 2022.09: [**Mixed-Precision Quantization Search**](configs/quant/README.md) is now supported! The [**QE-Score**](https://openreview.net/pdf?id=E28hy5isRzC) paper is accepted by NeurIPS'22.
- :boom: 2022.08: We will give a tutorial on [**Functional View for Zero-Shot NAS**](https://mlsys.org/virtual/2022/tutorial/2201) at MLSys'22.
- :boom: 2022.06: Code for [**MAE-DET**](configs/detection/README.md) is now released.
- :boom: 2022.05: [**MAE-DET**](https://proceedings.mlr.press/v162/sun22c/sun22c.pdf) is accepted by ICML'22.
- :boom: 2021.09: Code for [**Zen-NAS**](https://github.com/idstcv/ZenNAS) is now released.
- :boom: 2021.07: The inspiring training-free paper [**Zen-NAS**](https://openaccess.thecvf.com/content/ICCV2021/papers/Lin_Zen-NAS_A_Zero-Shot_NAS_for_High-Performance_Image_Recognition_ICCV_2021_paper.pdf) has been accepted by ICCV'21.

## Features

- This toolbox consists of multiple modules including the following :
    - [Search Searcher module](tinynas/searchers/README.md)
    - [Search Strategy module](tinynas/strategy/README.md)
    - [Models Definition module](tinynas/models/README.md)
    - [Score module](tinynas/scores/README.md)
    - [Search Space module](tinynas/spaces/README.md)
    - [Budgets module](tinynas/budgets/README.md)
    - [Latency Module](tinynas/latency/op_profiler/README.md)
    - [Population module](tinynas/evolutions/README.md)

It manages these modules with the help of [ModelScope](https://github.com/modelscope/modelscope) Registry and Configuration mechanism.

- The `Searcher` is defined to be responsible for building and completing the entire search process. Through the combination of these modules and the corresponding configuration files, we can complete backbone search for different tasks (such as classification, detection, etc.) under different budget constraints (such as the number of parameters, FLOPs, delay, etc.).

- Currently supported tasks: For each task, we provide several sample configurations and scripts as follows to help you get started quickly.

    - `Classification`：Please Refer to [Search Space](tinynas/spaces/space_k1kxk1.py) and [Config](configs/classification/R50_FLOPs.py)
    - `Detection`：Please Refer to [Search Space](tinynas/spaces/space_k1kx.py) and [Config](configs/detection/R50_FLOPs.py)
    - `Quantization`: Please Refer to [Search Space](tinynas/spaces/space_quant_k1dwk1.py) and [Config](configs/quant/Mixed_7d0G.py)

***
## Installation
- Please Refer to [installation.md](installation.md)

***
## How to Use
- Please Refer to [get_started.md](get_started.md)

***
## Results
### Results for Classification（[Details](configs/classification/README.md)）

|Backbone|Param (MB)|FLOPs (G)|ImageNet TOP1|Structure|Download|
|:----|:----|:----|:----|:----|:----|
|DeepMAD-R18|11.69|1.82|77.7%| [txt](configs/classification/models/deepmad-R18.txt)|[model](http://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DeepMAD/DeepMAD-R18/R18.pth.tar)|
|DeepMAD-R34|21.80|3.68|79.7%| [txt](configs/classification/models/deepmad-R34.txt)|[model](http://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DeepMAD/DeepMAD-R18/R34.pth.tar) |
|DeepMAD-R50|25.55|4.13|80.6%|[txt](configs/classification/models/deepmad-R50.txt) |[model](http://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DeepMAD/DeepMAD-R18/R50.pth.tar) |
|DeepMAD-29M-224|29|4.5|82.5%|[txt](configs/classification/models/deepmad-29M-224.txt) |[model](http://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DeepMAD/DeepMAD-29M-224/DeepMAD-29M-Res224-82.5acc.pth.tar) |
|DeepMAD-29M-288|29|4.5|82.8%|[txt](configs/classification/models/deepmad-29M-288.txt) |[model](http://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DeepMAD/DeepMAD-29M-288/DeepMAD-29M-Res288-82.8acc.pth.tar) |
|DeepMAD-50M|50|8.7|83.9%|[txt](configs/classification/models/deepmad-50M.txt) |[model](http://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DeepMAD/DeepMAD-50M/DeepMAD-50M-Res224-83.9acc.pth.tar) |
|DeepMAD-89M|89|15.4|84.0%|[txt](configs/classification/models/deepmad-89M.txt) |[model](http://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DeepMAD/DeepMAD-89M/DeepMAD-89M-Res224-84.0acc.pth.tar) |                                                   
| Zen-NAS-R18-like | 10.8 |    1.7     |   78.44  | [txt](configs/classification/models/R18-like.txt)       |[model](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/LightNAS/classfication/R18-like.pth.tar) |
| Zen-NAS-R50-like | 21.3 |    3.6     |   80.04  | [txt](configs/classification/models/R50-like.txt)       |[model](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/LightNAS/classfication/R50-like.pth.tar) |
| Zen-NAS-R152-like | 53.5 |    10.5     |   81.59  | [txt](configs/classification/models/R152-like.txt)       |[model](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/LightNAS/classfication/R152-like.pth.tar) |
> The official code for **Zen-NAS** was originally released at https://github.com/idstcv/ZenNAS.   <br/>
                                                                                                                        
***
### Results for low-precision backbones（[Details](configs/quant/README.md)）

|Backbone|Param (MB)|BitOps (G)|ImageNet TOP1|Structure|Download|
|:----|:----|:----|:----|:----|:----|
|MBV2-8bit|3.4|19.2|71.90%| -| -|
|MBV2-4bit|2.3|7|68.90%| -|- |
|Mixed19d2G|3.2|18.8|74.80%|[txt](configs/quant/models/mixed7d0G.txt) |[model](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/LightNAS/quant/mixed-7d0G/quant_238_70.7660.pth.tar) |
|Mixed7d0G|2.2|6.9|70.80%|[txt](configs/quant/models/mixed19d2G.txt) |[model](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/LightNAS/quant/mixed-19d2G/quant_237_74.8180.pth.tar) |
                                                                              
***
### Results for Object Detection（[Details](configs/detection/README.md)）
| Backbone | Param (M) | FLOPs (G) |   box AP<sub>val</sub> |   box AP<sub>S</sub> |   box AP<sub>M</sub>  |   box AP<sub>L</sub> | Structure | Download |
|:---------:|:---------:|:---------:|:-------:|:-------:|:-------:|:-------:|:--------:|:------:|
| ResNet-50 | 23.5 |    83.6    |  44.7 | 29.1 | 48.1 | 56.6  | - | - |
| ResNet-101| 42.4 |    159.5   |  46.3 | 29.9 | 50.1 | 58.7  | - | - |
| MAE-DET-S | 21.2 |    48.7    |  45.1 | 27.9 | 49.1 | 58.0  | [txt](configs/detection/models/maedet_s.txt)       |[model](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/LightNAS/detection/maedet-s/latest.pth) |
| MAE-DET-M | 25.8 |    89.9    |  46.9 | 30.1 | 50.9 | 59.9  | [txt](configs/detection/models/maedet_m.txt)       |[model](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/LightNAS/detection/maedet-m/latest.pth) |
| MAE-DET-L | 43.9 |    152.9   |  47.8 | 30.3 | 51.9 | 61.1  | [txt](configs/detection/models/maedet_l.txt)      |[model](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/LightNAS/detection/maedet-l/latest.pth) |

***
## Results for Action Recognition ([Details](configs/action_recognition/README.md)）

| Backbone  | size   |  FLOPs (G) |  SSV1 Top-1 | SSV1 Top-5 | Structure | 
|:---------:|:-------:|:-------:|:-------:|:-------:|:--------:|
| X3D-S | 160 |    1.9     |   44.6  | 74.4| -     |
| X3D-S | 224 |    1.9     |   47.3  | 76.6| -     |
| E3D-S | 160 |    1.9     |   47.1  | 75.6| [txt](configs/action_recognition/models/E3D_S.txt)       |
| E3D-M  | 224 |     4.7     |   49.4  | 78.1| [txt](configs/action_recognition/models/E3D_M.txt)       |
| E3D-L  | 312 |     18.3     |   51.1  | 78.7| [txt](configs/action_recognition/models/E3D_L.txt)       |

***
**Note**：
If you find this useful, please support us by citing them.
```
@inproceedings{cvpr2023deepmad,
	title = {DeepMAD: Mathematical Architecture Design for Deep Convolutional Neural Network},
	author = {Xuan Shen and Yaohua Wang and Ming Lin and Yilun Huang and Hao Tang and Xiuyu Sun and Yanzhi Wang},
	booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
	year = {2023},
	url = {https://arxiv.org/abs/2303.02165}
}

@inproceedings{icml23prenas,
	title={PreNAS: Preferred One-Shot Learning Towards Efficient Neural Architecture Search},
	author={Haibin Wang and Ce Ge and Hesen Chen and Xiuyu Sun},
	booktitle={International Conference on Machine Learning},
	year={2023},
	organization={PMLR}
}

@inproceedings{iclr23maxste,
	title     = {Maximizing Spatio-Temporal Entropy of Deep 3D CNNs for Efficient Video Recognition},
	author    = {Junyan Wang and Zhenhong Sun and Yichen Qian and Dong Gong and Xiuyu Sun and Ming Lin and Maurice Pagnucco and Yang Song },
	journal   = {International Conference on Learning Representations},
	year      = {2023},
}

@inproceedings{neurips23qescore,
	title     = {Entropy-Driven Mixed-Precision Quantization for Deep Network Design},
	author    = {Zhenhong Sun and Ce Ge and Junyan Wang and Ming Lin and Hesen Chen and Hao Li and Xiuyu Sun},
	journal   = {Advances in Neural Information Processing Systems},
	year      = {2022},
}

@inproceedings{icml22maedet,
	title={MAE-DET: Revisiting Maximum Entropy Principle in Zero-Shot NAS for Efficient Object Detection},
	author={Zhenhong Sun and Ming Lin and Xiuyu Sun and Zhiyu Tan and Hao Li and Rong Jin},
	booktitle={International Conference on Machine Learning},
	year={2022},
	organization={PMLR}
}

@inproceedings{iccv21zennas,
	title     = {Zen-NAS: A Zero-Shot NAS for High-Performance Deep Image Recognition},
	author    = {Ming Lin and Pichao Wang and Zhenhong Sun and Hesen Chen and Xiuyu Sun and Qi Qian and Hao Li and Rong Jin},
	booktitle = {2021 IEEE/CVF International Conference on Computer Vision},
	year      = {2021},
}
```
                                                                                                                           
## License

This project is developed by Alibaba and licensed under the [Apache 2.0 license](LICENSE).

This product contains third-party components under other open source licenses.

See the [NOTICE](NOTICE) file for more information.
=======
# Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection

GFocalV2 (GFLV2) is a next generation of GFocalV1 (GFLV1), which utilizes the statistics of learned bounding box distributions to guide the reliable localization quality estimation.

Again, GFLV2 improves over GFLV1 about ~1 AP without (almost) extra computing cost! Analysis of GFocalV2 in ZhiHu: [大白话 Generalized Focal Loss V2](https://zhuanlan.zhihu.com/p/313684358). You can see more comments about GFocalV1 in [大白话 Generalized Focal Loss(知乎)](https://zhuanlan.zhihu.com/p/147691786)

---

More news:

[2021.3] GFocalV2 has been accepted by CVPR2021 (pre-review score: 113).

[2020.11] GFocalV1 has been adopted in [NanoDet](https://github.com/RangiLyu/nanodet), a super efficient object detector on mobile devices, achieving same performance but 2x faster than YoLoV4-Tiny! More details are in [YOLO之外的另一选择，手机端97FPS的Anchor-Free目标检测模型NanoDet现已开源~](https://zhuanlan.zhihu.com/p/306530300).

[2020.10] Good News! GFocalV1 has been accepted in NeurIPs 2020 and GFocalV2 is on the way.

[2020.9] The winner (1st) of GigaVision (object detection and tracking) in ECCV 2020 workshop from DeepBlueAI team adopt GFocalV1 in their [solutions](http://dy.163.com/article/FLF2LGTP0511ABV6.html).

[2020.7] GFocalV1 is officially included in [MMDetection V2](https://github.com/open-mmlab/mmdetection/blob/master/configs/gfl/README.md), many thanks to [@ZwwWayne](https://github.com/ZwwWayne) and [@hellock](https://github.com/hellock) for helping migrating the code.


## Introduction

Localization Quality Estimation (LQE) is crucial and popular in the recent advancement of dense object detectors since it can provide accurate ranking scores that benefit the Non-Maximum Suppression processing and improve detection performance.  As a common practice, most existing methods predict LQE scores through vanilla convolutional features shared with object classification or bounding box regression. In this paper, we explore a completely novel and different perspective to perform LQE -- based on the learned distributions of the four parameters of the bounding box. The bounding box distributions are inspired and introduced as ''General Distribution'' in GFLV1, which describes the uncertainty of the predicted bounding boxes well. Such a property makes the distribution statistics of a bounding box highly correlated to its real localization quality. Specifically, a bounding box distribution with a sharp peak usually corresponds to high localization quality, and vice versa. By leveraging the close correlation between distribution statistics and the real localization quality, we develop a considerably lightweight Distribution-Guided Quality Predictor (DGQP) for reliable LQE based on GFLV1, thus producing GFLV2. To our best knowledge, it is the first attempt in object detection to use a highly relevant, statistical representation to facilitate LQE. Extensive experiments demonstrate the effectiveness of our method. Notably, GFLV2 (ResNet-101) achieves 46.2 AP at 14.6 FPS, surpassing the previous state-of-the-art ATSS baseline (43.6 AP at 14.6 FPS) by absolute 2.6 AP on COCO test-dev, without sacrificing the efficiency both in training and inference.

<img src="https://github.com/implus/GFocalV2/blob/master/gfocal.png" width="1000" height="300" align="middle"/>

For details see [GFocalV2](https://arxiv.org/pdf/2011.12885.pdf). The speed-accuracy trade-off is as follows:

<img src="https://github.com/implus/GFocalV2/blob/master/sota_time_acc.jpg" width="541" height="365" align="middle"/>


## Get Started

Please see [GETTING_STARTED.md](https://github.com/open-mmlab/mmdetection/blob/v2.6.0/docs/get_started.md) for the basic usage of MMDetection.

## Train

```python
# assume that you are under the root directory of this project,
# and you have activated your virtual environment if needed.
# and with COCO datas in 'datas/coco/'

./tools/dist_train.sh configs/gfocal/gfocal_r50_fpn_ms2x.py 8 --validate
```

## Inference

```python
./tools/dist_test.sh configs/gfocal/gfocal_r50_fpn_ms2x.py work_dirs/gfocal_r50_fpn_ms2x/epoch_24.pth 8 --eval bbox
```

## Speed Test (FPS)

```python
CUDA_VISIBLE_DEVICES=0 python3 ./tools/benchmark.py configs/gfocal/gfocal_r50_fpn_ms2x.py work_dirs/gfocal_r50_fpn_ms2x/epoch_24.pth
```

## Models

For your convenience, we provide the following trained models (GFocalV2). All models are trained with 16 images in a mini-batch with 8 GPUs.

Model | Multi-scale training | AP (minival) | AP (test-dev) | FPS | Link
--- |:---:|:---:|:---:|:---:|:---:
GFocal_R_50_FPN_1x              | No  | 41.0 | 41.1 | 19.4 | [Google](https://drive.google.com/file/d/1wSE9-c7tcQwIDPC6Vm_yfOokdPfmYmy7/view?usp=sharing)
GFocal_R_50_FPN_2x              | Yes | 43.9 | 44.4 | 19.4 | [Google](https://drive.google.com/file/d/17-1cKRdR5J3SfZ9NBCwe6QE554uTS30F/view?usp=sharing)
GFocal_R_101_FPN_2x             | Yes | 45.8 | 46.0 | 14.6 | [Google](https://drive.google.com/file/d/1qomgA7mzKW0bwybtG4Avqahv67FUxmNx/view?usp=sharing)
GFocal_R_101_dcnv2_FPN_2x       | Yes | 48.0 | 48.2 | 12.7 | [Google](https://drive.google.com/file/d/1xsBjxmqsJoYZYPMr0k06X5K9nnPrexcx/view?usp=sharing)
GFocal_X_101_dcnv2_FPN_2x       | Yes | 48.8 | 49.0 | 10.7 | [Google](https://drive.google.com/file/d/1AHDVQoclYPSP0Ync2a5FCsr_rhq2QdMH/view?usp=sharing)
GFocal_R2_101_dcnv2_FPN_2x      | Yes | 49.9 | 50.5 | 10.9 | [Google](https://drive.google.com/file/d/1sAXfYLXIxZgMrC44LBqDgfYImThZ_kud/view?usp=sharing)

[0] *The reported numbers here are from new experimental trials (in the cleaned repo), which may be slightly different from the original paper.* \
[1] *Note that the 1x performance may be slightly unstable due to insufficient training. In practice, the 2x results are considerably stable between multiple runs.* \
[2] *All results are obtained with a single model and without any test time data augmentation such as multi-scale, flipping and etc..* \
[3] *`dcnv2` denotes deformable convolutional networks v2. Note that for ResNe(X)t based models, we apply deformable convolutions from stage c3 to c5 in backbones.* \
[4] *Refer to more details in config files in `config/gfocal/`.* \
[5] *FPS is tested with a single GeForce RTX 2080Ti GPU, using a batch size of 1.* 


## Acknowledgement

Thanks MMDetection team for the wonderful open source project!


## Citation

If you find GFocal useful in your research, please consider citing:

```
@article{li2020gfl,
  title={Generalized focal loss: Learning qualified and distributed bounding boxes for dense object detection},
  author={Li, Xiang and Wang, Wenhai and Wu, Lijun and Chen, Shuo and Hu, Xiaolin and Li, Jun and Tang, Jinhui and Yang, Jian},
  journal={arXiv preprint arXiv:2006.04388},
  year={2020}
}
```

```
@article{li2020gflv2,
  title={Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection},
  author={Li, Xiang and Wang, Wenhai and Hu, Xiaolin and Li, Jun and Tang, Jinhui and Yang, Jian},
  journal={arXiv preprint arXiv:2011.12885},
  year={2020}
}
```

>>>>>>> origin/master


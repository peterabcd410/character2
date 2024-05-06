# 概述

本章的代码基于mmdetection，基准模型为RetinaNet。

参考链接：https://mmdetection.readthedocs.io/zh-cn/latest/user_guides/index.html

# 命令

创建并激活一个 conda 环境。

```
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

基于 [PyTorch 官方说明](https://pytorch.org/get-started/locally/)安装 PyTorch。

```
conda install pytorch torchvision -c pytorch
```

使用 [MIM](https://github.com/open-mmlab/mim) 安装 [MMEngine](https://github.com/open-mmlab/mmengine) 和 [MMCV](https://github.com/open-mmlab/mmcv)。

```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

安装环境

```
pip install -v -e .
```

训练

```
python tools/train.py configs/rtmdet_tiny_8xb32-300e_coco.py
```


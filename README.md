# Faster_RCNN

> 根据Faster RCNN论文以及其他开源代码完成的基于`pytorch1.0`的复现代码
## Introduction
- 增加了许多的代码注释
- 简化了许多过程，适合学习使用
- 所有的代码都是pytorch写的，没有使用C编译的代码

## DataSet
1. 下载VOC2007数据集
```shell
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
```
2. 解压至`data/`文件夹, 或使用软连接
```
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar
```

## Train
1. 安装所需的python依赖

2. 训练的各项参数设置都在`model_train.py`文件中，没有设置命令行输入

    修改完超参数后直接运行训练程序即可：
    ```
    python model_train.py
    ```

关于`nms`部分，没有使用GPU加速，始终使用的都是CPU。



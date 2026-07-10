# MNIST

手写数字识别


## 数据集

MNIST 数据集下载地址:
wget https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz

## 精度

98.88%

## 使用说明

$ mkdir -p ./save_model

启动训练脚本：python train.py

启动验证脚本：python eval.py

## 训练参数文件

训练获得的网络参数保存在 save_model 文件夹下, 验证时需调用 save_model 文件夹下的网络参数文件。



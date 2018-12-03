# PyTorch Benchmark

## Environment

* torch version: 1.0.0a0+ff608a9
* CUDA version: 10.0.130
* DISTRIB_DESCRIPTION="Ubuntu 16.04 LTS"

## Layer config

### Linear

* Input shape: torch.Size([1, 4096, 4096])
* Op: Linear(in_features=4096, out_features=4096, bias=False)
* 128.000GFLOP

### Conv2d

* Input shape: torch.Size([1, 256, 256, 256])
* Op: Conv2d(256, 256, kernel_size=(8, 8), stride=(1, 1), bias=False)
* 484.383GFLOP

## Result

|   GPU Model | FP32 Linear | FP16 Linear | Linear Ratio | FP32 Conv2d | FP16 Conv2d | Conv2d Ratio |
| ----------- | ----------- | ----------- | ------------ | ----------- | ----------- | ------------ |
|     TITAN V |  12.9TFLOPS |  77.4TFLOPS |      598.61% |  8.71TFLOPS |  76.4TFLOPS |      875.61% |
| RTX 2080 Ti |  12.7TFLOPS |  53.4TFLOPS |      419.31% |  6.20TFLOPS |  51.9TFLOPS |      750.22% |
| RTX 2080    |  9.06TFLOPS |  36.2TFLOPS |      398.94% |  4.95TFLOPS |  35.8TFLOPS |      721.24% |
| GTX 1080 Ti |  10.5TFLOPS |  10.3TFLOPS |       98.74% |  6.29TFLOPS |  7.14TFLOPS |      108.55% |

## 编译 PyTorch

参考链接：https://github.com/pytorch/pytorch#from-source

### 准备工作

* Ubuntu 16.04.5
* CUDA cuda_10.0.130_410.48_linux.run
* cuDNN cudnn-10.0-linux-x64-v7.4.1.5
* Anaconda 5.2

#### CUDA

下载地址：https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=runfilelocal

直接使用下面的命令即可安装完成：

```sh
sudo bash cuda_10.0.130_410.48_linux.run
```

如果你之前安装过其他版本的 CUDA，并且你没有让安装包自动链接 `/usr/local/cuda`，那么你需要执行下面的命令：

```sh
sudo rm /usr/local/cuda && sudo ln -s /usr/local/cuda-10.0/ /usr/local/cuda
```

#### cuDNN

下载地址：https://developer.nvidia.com/rdp/cudnn-download

解压并移动到 `/usr/local/cuda-10.0/` 目录下即可：

```sh
tar -xzvf cudnn-10.0-linux-x64-v7.4.1.5.tgz
sudo mv cuda/include/* /usr/local/cuda-10.0/include
sudo cp cuda/lib64/* /usr/local/cuda-10.0/lib64
```

#### Anaconda

下载地址：https://www.anaconda.com/download/#linux

清华镜像：https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/

Anaconda 直接运行即可：

```sh
bash Anaconda3-5.2.0-Linux-x86_64.sh
```

### 安装 Anaconda 依赖项

```sh
conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing
conda install -c mingfeima mkldnn
```

### 复制 PyTorch 代码库

#### git clone

复制 PyTorch 代码库：

```sh
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
```

Tips：如果你忘记使用 `--recursive` 了，可以使用下面的命令重新下载 PyTorch 依赖的代码库：

```sh
git submodule update --init --recursive
```

#### git checkout

切到 v1.0rc1：

```sh
git checkout ff608a9ff3edded33764c8631427e92c7288bafb
```

参考链接：https://github.com/pytorch/pytorch/releases/tag/v1.0rc1

### 编译 PyTorch

#### CMAKE_PREFIX_PATH

首先你需要设置 conda 路径，如果你使用的是 Anaconda 5.2，那么可以通过下面的命令设置：

```sh
export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
```

否者你需要手动设置：

```sh
export CMAKE_PREFIX_PATH=/home/ypw/anaconda3/bin
```

#### build

现在可以通过下面的命令编译 PyTorch 了：

```sh
python setup.py bdist_wheel
```

编译完成之后，安装包会在 dist 目录下，你可以通过下面的命令安装：

```sh
pip uninstall torch
pip install dist/torch-1.0.0a0+5c89190-cp36-cp36m-linux_x86_64.whl
```

### 疑难杂症

如果你之前在主分支下编译过 PyTorch，并且不能用，你可能需要先执行下面的命令：

```sh
python setup.py clean --all
```

### 测试

#### CUDA 版本测试

你可以使用下面的命令测试 PyTorch 是否使用 CUDA10：

```sh
python -c "import torch; \
print('torch version:', torch.__version__);\
print('CUDA version:', torch.version.cuda);\
print('CUDNN version:', torch.backends.cudnn.version())"
```

它应该输出：

```
torch version: 1.0.0a0+5c89190
CUDA version: 10.0.130
CUDNN version: 7401
```

#### 矩阵乘法测试

你可以使用下面的命令测试 PyTorch 是否能够使用显卡进行矩阵乘法：

```sh
python -c "import torch; \
X = torch.randn(1024, 1024).half().cuda(); \
B = torch.matmul(X, X); \
print(B.shape)"
```

它应该输出 `torch.Size([1024, 1024])`。

#### 注意：不要在 pytorch 目录下测试

不然 `import torch` 会出错：

```py
>>> import torch
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/ypw/pytorch/torch/__init__.py", line 84, in <module>
    from torch._C import *
ModuleNotFoundError: No module named 'torch._C'
```

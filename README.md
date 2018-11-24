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

# SSPQ
Spatial Shift Point-Wise Quantization code open

# introduction
This repository is related with the paper Spatial Shift Point-Wise Quantization.
The main idea is
 - substitute the spatial operation to active shift operation
 - apply inverted residual block structure as introduced by the paper all you need is a few shifts - https://arxiv.org/abs/1903.05285 .
 - quantize 1x1 point-wise convolution part by using dorefa-net.
 - keep the quantization bit for activation over 4-bit since quantization under 4-bit occur unstablility of convergence.
 
# our experimental Results for SSPQ 50 
This figure shows the SSPQ 50 light-weight model performance in the metric of accuracy and model size.
![performance](https://github.com/Eunhui-Kim/SSPQ/blob/main/Fig4-s.png)

 
# pre-requisite
Note that this code is tested only in the environment decribed below. Mismatched versions does not guarantee correct execution.

 - Ubuntu kernel ver. 4.15.0-117-generic #118~16.04.1
 - Cuda 10.0
 - cudnn 7.6.5
 - Tensorflow 1.15.3
 - Tensorpack 
 - g++ 7.5.0
 - python 3.7
 - install https://github.com/Eunhui-Kim/custom-op-for-shiftNet
   
   After the test works fine, at first 
   
 - put the shift2d.py in the tensorpack models path.
   in my case, the tensorpack models is
   $HOME/.local/lib/python3.7/site-packages/tensorpack/models/

# Testing
  sspq50 model for imagenet dataset, 
  1) put 'imagenet-resnet-QresIBShift.py, resnet_model2.py, with imagenet_utils.py and dorefa.py of tensorpack file together in the tensorpack example path
  2) run
  python imagenet-resnet-QresIBShift.py --mode QresIBShift --data $Imagenet_Path --batch 256 --depth 50    
   
  

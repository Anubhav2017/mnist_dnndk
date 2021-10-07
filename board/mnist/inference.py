import sys
sys.path.append('/usr/local/lib/python2.7/dist-packages')
import dnndk
from dnndk import n2cube, dputils
from ctypes import *
import cv2
import numpy as np
import os
import threading
import time
import sys
from matplotlib import pyplot as plt
import matplotlib

KERNEL_CONV = "mnist"
KERNEL_CONV_INPUT = "conv2d_Conv2D(0)"
KERNEL_FC_OUTPUT = "dense_1_MatMul(0)"

def preprocess_fn(image_path):
    '''
    Image pre-processing.
    Opens image as grayscale then normalizes to range 0:1
    input arg: path of image file
    return: numpy array
    '''
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = image.reshape(28,28,1)
    image = image/255.0
    return image

image_dir="test_images"
listimage=os.listdir(image_dir)
runTotal = len(listimage)
               
imgs = []
for i in range(runTotal):
    path = os.path.join(image_dir,listimage[i])
    imgs.append(preprocess_fn(path))

# Attach to DPU driver and prepare for runing
n2cube.dpuOpen()
# Create DPU Kernels for ResNet50
kernel = n2cube.dpuLoadKernel(KERNEL_CONV)
# Create DPU Tasks from DPU Kernel
task = n2cube.dpuCreateTask(kernel, 0)

size = n2cube.dpuGetInputTensorSize(task, KERNEL_CONV_INPUT)
# Get the output tensor channel from FC output
channel = n2cube.dpuGetOutputTensorSize(task, KERNEL_FC_OUTPUT)

FCResult = [0 for i in range(channel)]
out_q=[]

t1=time.time()
for img in imgs:
    n2cube.dpuSetInputTensorInHWCFP32(task, KERNEL_CONV_INPUT,img.reshape(size), size)
    n2cube.dpuRunTask(task)
    n2cube.dpuGetOutputTensorInHWCFP32(task, KERNEL_FC_OUTPUT, FCResult, channel)
    out_q.append(FCResult.index(max(FCResult)))

t2=time.time()

print "inference time=", t2-t1

classes = ['zero','one','two','three','four','five','six','seven','eight','nine'] 
correct = 0
wrong = 0
for i in range(len(out_q)):
    prediction = classes[out_q[i]]
    ground_truth, _ = listimage[i].split('_',1)
    if (ground_truth==prediction):
        correct += 1
    else:
        wrong += 1
accuracy = correct/len(out_q)
print 'Correct:%d, Wrong:%d, Accuracy:%.4f' %(correct,wrong,accuracy)

rtn = n2cube.dpuDestroyKernel(kernel)
n2cube.dpuClose()
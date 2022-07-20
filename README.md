# COVID-WideNet—A capsule network for COVID-19 detection
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg) 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/covid-widenet-a-capsule-network-for-covid-19/covid-19-diagnosis-on-covid-ct)](https://paperswithcode.com/sota/covid-19-diagnosis-on-covid-ct?p=covid-widenet-a-capsule-network-for-covid-19)

Abstract - In this paper, we propose a capsule network called COVID-WideNet for diagnosing COVID-19 cases using Chest X-ray (CXR) images. Experimental results have demonstrated that a discriminative trained, multi-layer capsule network achieves state-of-the-art performance on the COVIDx dataset. In particular, COVID-WideNet performs better than any other CNN based approaches for the diagnosis of COVID-19 infected patients. Further, the proposed COVID-WideNet has the number of trainable parameters that is 20 times less than that of other CNN based models. This results in a fast and efficient diagnosing COVID-19 symptoms, and with achieving the 0.95 of Area Under Curve (AUC), 91% of accuracy, sensitivity and specificity, respectively. This may also assist radiologists to detect COVID and its variant like delta.

![alt text](https://ars.els-cdn.com/content/image/1-s2.0-S1568494622002046-gr2_lrg.jpg)

Official Implementation of the paper - [COVID-WideNet—A capsule network for COVID-19 detection](https://www.sciencedirect.com/science/article/pii/S1568494622002046)

capsulelayers.py taken from https://github.com/XifengGuo/CapsNet-Keras/blob/master/capsulelayers.py \
train.py taken from https://github.com/changspencer/Tumor-CapsNet/issues/3 \
preprocess.py taken from https://github.com/ShahinSHH/COVID-CAPS 

dataset taken from https://github.com/UCSD-AI4H/COVID-CT 

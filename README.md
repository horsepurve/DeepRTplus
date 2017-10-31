# DeepRT: deep learning for peptide retention time prediction in proteomics

## Preprint
[arXiv:1705.05368](https://arxiv.org/abs/1705.05368)

## Introduction
In separation science and analytical chemistry, the prediction of retention times were primarily based on retention coefficients of amino acids or molecular descriptors of metabolites. 

DeepRT is an software for peptide retention prediction, developed collaboratively by BGI-Shenzhen and Intel China, outperforming other software such as ELUDE, GPTime, etc. The algorithm is based on the state-of-the-art deep learning algorithm residual network (ResNet) and long short-time memory (LSTM), written based upon the efficient deep learning library MXNet (https://github.com/dmlc/mxnet). If you use DeepRT in your research, please cite: https://arxiv.org/abs/1705.05368. Currently only the Windows 64 bit version binary package is available. If you have any question, or need other version of DeepRT (Windows 32 bit, Linux, GPU), please contact me (machunwei@genomics.cn).

## Usage
First fill in the configuration file config.json, in which all parameters for the software are stored. If you have not split the dataset into training and testing data, you can use the tool in the package, and if you have your own training and testing datasets, you can just skip this step:
```
data_split.exe config.json
```
To train the LSTM network, run as:
```
lstm_train.exe config.json
```
To see the result of each epoch of LSTM:
```
performance_monitor.exe config.json
```
For ResNet training, run as:
```
resnet_train.exe config.json
```
After the training processes of LSTM and ResNet are all finished, we can ensemble the results:
```
easy_ensemble.exe config_mod.json
```
And then the Pearson correlation, RMSE, &Delta; t<sub>95%</sub> and the running time will be reported. The predicted retention time for each peptide in the testing dataset will be written to the results directory.

## contact
machunwei@genomics.cn

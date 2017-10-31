# DeepRT: deep learning for peptide retention time prediction in proteomics

## Preprint
[arXiv:1705.05368](https://arxiv.org/abs/1705.05368)

## Introduction
In separation science and analytical chemistry, the prediction of retention times were primarily based on retention coefficients of amino acids or molecular descriptors of metabolites. Traditionally, retention times of peptides with or without post-translational modifications are predicted separately, and retention times in RPLC, HILIC or SCX are also predicted separately. DeepRT, instead, provides a generic framework for RT prediction, whose architecture are shown as following figure.
 <div align="center"><img src="https://github.com/horsepurve/DeepRT/blob/master/img/figure_1.png" width="45%" alt="figure_1" /></div>

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
And then the Pearson correlation, RMSE, &Delta;t<sub>95%</sub> and the running time will be reported. The predicted retention time for each peptide in the testing dataset will be written to the results directory.

## contact
machunwei@genomics.cn

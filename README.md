# DeepRT: Deep Transfer Learning for Peptide Retention Time Prediction under Different Liquid Chromatographic Conditions

<!-- ## Preprint 
[arXiv:1705.05368](https://arxiv.org/abs/1705.05368) (15 May 2017)
</br>[arXiv:1711.00045](https://arxiv.org/abs/1711.00045) (31 Oct 2017) -->
## Citation
@article{ma2017retention,
</br>&nbsp;&nbsp;title={Retention Time of Peptides in Liquid Chromatography Is Well Estimated upon Deep Transfer Learning},
</br>&nbsp;&nbsp;author={Ma, Chunwei and Zhu, Zhiyong and Ye, Jun and Yang, Jiarui and Pei, Jianguo and Xu, Shaohang and Yu, Chang and Mo, Fan and Wen, Bo and Liu, Siqi},
</br>&nbsp;&nbsp;journal={arXiv preprint [arXiv:1711.00045](https://arxiv.org/abs/1711.00045)},
</br>&nbsp;&nbsp;year={2017}
</br>}

## Brief Introduction
In separation science and analytical chemistry, the predictions of retention times were primarily based on retention coefficients of amino acids or molecular descriptors of metabolites. Traditionally, retention times of peptides with or without post-translational modifications are predicted separately, and retention times in RPLC, HILIC or SCX are also predicted separately. DeepRT, instead, provides a generic framework for RT prediction, whose architecture is shown in the following figure.
 <div align="center"><img src="https://github.com/horsepurve/DeepRT/blob/master/img/figure_1.png" width="45%" alt="figure_1" /></div>
We adopted an N-to-1 LSTM architecture for RT modeling while ResNet was upon the one-hot encoded peptides, in parallel. Finally, the two outputs were assembled.

## Prediction Accuracy
Using a large peptide dataset generated from RPLC with 146587 peptides (131928 peptides for training and 14659 for testing), DeepRT achieved a **Pearson's correlation as high as 0.996** and an <b>R<sup>2</sup> of 0.993</b>, as shown in following figure.
<div align="center"><img src="https://github.com/horsepurve/DeepRT/blob/master/img/figure_4.png" width="60%" alt="figure_1" /></div>
While transferring this trained model to other datasets, the prediction accuracy was further improved and the running time was reduced, even though the source and target datasets were generated under different liquid chromatographic conditions, as shown in the figure below, in which (A, B, C, D) are of modified dataset while (E, F, G, H) are of unmodified dataset.
<div align="center"><img src="https://github.com/horsepurve/DeepRT/blob/master/img/figure_2.png" width="50%" alt="figure_1" /></div>
As a brief conclusion, by virtue of deep transfer learning, we can 1) predict RT for both modified and unmodified peptides using the same algorithm, 2) refine RT estimation using pretrained RT model, 3) use unmodified peptides to help prediction of modified peptides and vice versa.
</br></br>Recent studies of this year in analytical chemistry exhibited that the accuracy of relatively simple additive models for RT prediction decreases in the order: CZE (0.995 R<sup>2</sup>) > SCX (0.991 R<sup>2</sup>) > HILIC (0.98 R<sup>2</sup>) > RPLC (∼0.965 R<sup>2</sup>), because of the difference in their separation mechanisms. DeepRT improves the accuracy of RT prediction in RPLC up to as high as ~0.993 R<sup>2</sup>, approaching that of CZE, and thus gives separation scientists insights into the selection of LC types in LC-MS experiments.

## Data Efficiency and Time Complexity
The following figure shows the performance of DeepRT with training data incresing. While testing, ResNet was run on NVIDIA Tesla M2070 while LSTM was run on Intel CPU with 12 cores. With training data larger than 10k peptides, SVM-based method was prohibitively slow while DeepRT, however, was still efficient, due to its linear time complexity w.r.t number of samples.
<div align="center"><img src="https://github.com/horsepurve/DeepRT/blob/master/img/figure_3.png" width="60%" alt="figure_1" /></div>

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

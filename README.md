# DeepRT(+): ultra-precise peptide retention predictor

* [1 Installation](#1) 
* [2 Scripts to reproduce the results](#2)
    - [2.1 RPLC datasets](#2.1)
    - [2.2 Other datasets](#2.2)
* [3 Change to your own datasets](#3)
    - [3.1 Datasets](#3.1)
    - [3.2 Model parameters](#3.2)
* [4 Transfer learning using our pre-trained models](#4)    
* [5 Make prediction using the trained models](#5)    
* [6 Citation](#6)    
* [7 Other models](#7)    
* [8 CPU version](#8)    
* [9 Questions](#9)    

<h2 id="1">1 Installation</h2>

```
git clone https://github.com/horsepurve/DeepRTplus  
cd DeepRTplus
```
And then follow DeepRT_install.sh to install the prerequisites.

<h2 id="2">2 Scripts to reproduce the results</h2>

<h3 id="2.1">2.1 RPLC datasets</h3>

Let's see how to apply DeepRT on HeLa dataset (modifications included). Simply type:

```
python data_split.py data/mod.txt 9 1 2
python capsule_network_emb.py
```

The HeLa data is split with 9:1 ratio with random seed 2, 9 for training and 1 for testing, and then capsule network begins training. You may check out the prediction result (about 0.985 ACC) and log file in typically 3 minutes (on a laptop with GTX 1070, for example).

To reproduce the result in the paper, just run as:

```
cd work
sh ../pipeline_mod.sh
```

And then you may see the reports in work directory.

<h3 id="2.2">2.2 Other datasets</h3>

See data/README_data.md for a summary and run corresponding pipline. All the necessary parameters for those datasets are stored in config_backup.py.

<h2 id="3">3 Change to your own datasets</h2>

<h3 id="3.1">3.1 Datasets</h3>

Prepare your dataset as the following format:

```
sequence	RT
4GSQEHPIGDK	2507.67
GDDLQAIK	2996.73
FA2FNAYENHLK	4681.428
AH3PLNTPDPSTK	2754.66
WDSE2NSERDVTK	2645.274
TEEGEIDY2AEEGENRR	3210.3959999999997
SQGD1QDLNGNNQSVTR	2468.946
```

Separate the peptide sequence and RT (in second) by tab (\t), encode the modified amino acides as digits (currently only four kinds of modification are included in the pre-trained models):

```
'M[16]' -> '1',
'S[80]' -> '2',
'T[80]' -> '3',
'Y[80]' -> '4'
```

<h3 id="3.2">3.2 Model parameters</h3>

There are only several parameters to specify for say HeLa data:

```
train_path = 'data/mod_train_2.txt' 
test_path = 'data/mod_test_2.txt' 
result_path = 'result/mod_test_2.pred.txt'
log_path = 'result/mod_test_2.log'
save_prefix = 'epochs'
pretrain_path = ''
dict_path = '' 

conv1_kernel = 10
conv2_kernel = 10

min_rt = 0
max_rt = 110 
time_scale = 60 
max_length = 50
```

<h2 id="4">4 Transfer learning using our pre-trained models</h2>

Training deep neural network models are time-consuming, especially for large dataset such as the Misc dataset here. However, the prediction accuracy is far from satisfactory without training dataset that big enough. The transfer leaning strategy used here can overcome this issue. You can use your small datasets in hand to fine-tune our pre-trained model in RPLC.

For a demo using the transfer learning strategy, just type:

```
cd work
sh ../pipeline_mod_trans_emb.sh
```

Note that you have to use the GPU version to load the pre-trained models, or otherwise you have to train from scratch on CPU.

<h2 id="5">5 Make prediction using the trained models</h2>

Predicting unknown RT for a new peptide using a current model is easy to do, see below as a demo, which is self-explained:

```
python prediction_emb.py 100 param/dia_all_trans_mod_epo20_dim24_conv10.pt 10 ${rt_file}
```

<h2 id="6">6 Publication</h2>

doi: 10.1021/acs.analchem.8b02386 ([PubMed](https://www.ncbi.nlm.nih.gov/pubmed/30114359))

<h2 id="7">7 Other models</h2>

As ResNet and LSTM (already been optimized) were less accurate then capsule network, the codes for ResNet and LSTM were deprecated, and DeepRT(+) (based on CapsNet) is recommended.

Of course you can still use SVM for training, use data_adaption.py to change the data format, and then import it to Elude/GPTime.

<h2 id="8">8 CPU version</h2>

Running DeepRT on CPU is not recommended, because it is way too slow. However, if you have to, duplicate capsule_network_emb.py to be capsule_network_emb_cpu.py and make a few amendments by yourself.

<h2 id="9">9 Questions</h2>

[contact](mailto:horsepurve@gmail.com)


# DeepRT(+): ultra-precise peptide retention predictor
Contents:
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
* [8 Questions](#8)    

<h2 id="1">1 Installation</h2>

```
git clone https://github.com/horsepurve/DeepRTplus  
```

<h2 id="2">2 Scripts to reproduce the results</h2>

<h3 id="2.1">2.1 RPLC datasets</h3>

Script to analyze the HeLa dataset (modification included).

```
sh pipeline_mod.sh  
```

<h3 id="2.2">2.2 Other datasets</h3>

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

Separate the peptide sequence and RT (in minute) by tab (\t), encode the modified amino acides as digits (currently only four kinds of modification are included in the pre-trained models):

```
'M[16]' -> '1',
'S[80]' -> '2',
'T[80]' -> '3',
'Y[80]' -> '4'
```

<h3 id="3.2">3.2 Model parameters</h3>

<h2 id="4">4 Transfer learning using our pre-trained models</h2>

Training deep neural network models are time-consuming, especially for large dataset such as the Misc dataset here. However, the prediction accuracy is far from satisfactory without training dataset that big enough. The transfer leaning strategy used here can overcome this issue. You can use your small datasets in hand to fine-tune our pre-trained model in RPLC.

Note that you have to use the GPU version to load the pre-trained models, or otherwise you have train from scratch on CPU.

<h2 id="5">5 Make prediction using the trained models</h2>

```
python prediction_emb.py 100 param/dia_all_trans_mod_epo20_dim24_conv10.pt 10 ${rt_file}
```

<h2 id="6">6 Citation</h2>

doi: 10.1021/acs.analchem.8b02386 ([PubMed](https://www.ncbi.nlm.nih.gov/pubmed/30114359))

<h2 id="7">7 Other models</h2>

As ResNet and LSTM (already been optimized) were less accurate then capsule network, the codes for ResNet and LSTM were deprecated, and DeepRT(+) (based on CapsNet) is recommended.

<h2 id="8">8 Question</h2>

[contact](mailto:horsepurve@gmail.com)


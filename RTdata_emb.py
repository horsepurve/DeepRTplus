import os
import numpy as np
import pandas as pd
import operator
from scipy import sparse
from config import *
import torch

class Dictionary(object):
    def __init__(self, path):
        self.word2idx = {}
        self.idx2word = []
        # for padding char:
        self.idx2word.append('*') # CNN_EMB
        self.word2idx['*'] = 0 # CNN_EMB
        self.build(path)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def build(self, path):
        assert os.path.exists(path)
        # construct the dictionary:        
        seq_data = pd.read_csv(path, sep = '\t')
        for aa in sorted(set(''.join(seq_data['sequence'].values))):
            self.add_word(aa)
        # print out the dictionary to see:
        for aa in sorted(self.word2idx.items(), key=operator.itemgetter(1)):
            print(aa[1],'->',aa[0])
        print('>> number of aa:', len(self.idx2word))

DATA_AUGMENTATION = False
class RTdata(object):
    def __init__(self, 
                 dictionary,
                 max_length, # we have to specify the max length in this version
                 path): # used to construct the dictionary                
        
        assert os.path.exists(path)        
    
        seq_data = pd.read_csv(path, sep = '\t')
        
        if False == DATA_AUGMENTATION:
            N_seq = len(seq_data['sequence'])
            self.number_seq = N_seq
            self.N_time_step = max_length # + 1 # +1 because we fellow ICLR2016attention paper
            self.N_aa = len(dictionary) 

            X = np.zeros((N_seq, self.N_aa, self.N_time_step))
            self.y = np.zeros(N_seq)
            # self.mask = np.zeros((N_seq, N_time_step), dtype=bool)

            # fill in the data:
            for seq_index, seq in enumerate(seq_data['sequence']):
                # fill in X, mask:
                for aa_index, aa in enumerate(seq):
                    X[seq_index, dictionary.word2idx[aa], aa_index] = 1                
                # fill in y:
                self.y[seq_index] = (seq_data.iloc[seq_index]['RT']/time_scale-min_rt)/(max_rt-min_rt) # second -> minute
                # self.y[seq_index] = len(seq) /50. # toy label
            print('>> note: using sparse matrix to store the data.')
            self.X = [sparse.csr_matrix(i) for i in X]
        
        if True == DATA_AUGMENTATION:
            N_seq = len(seq_data['sequence'])
            self.number_seq = N_seq
            self.N_time_step = max_length # + 1 # +1 because we fellow ICLR2016attention paper
            self.N_aa = len(dictionary) 

            X = np.zeros((N_seq*2, self.N_aa, self.N_time_step))
            self.y = np.zeros(N_seq*2)
            # self.mask = np.zeros((N_seq, N_time_step), dtype=bool)

            # fill in the data:
            for seq_index, seq in enumerate(seq_data['sequence']):
                # fill in X:
                for aa_index, aa in enumerate(seq):
                    X[seq_index, dictionary.word2idx[aa], aa_index] = 1                
                # fill in y:
                self.y[seq_index] = (seq_data.iloc[seq_index]['RT']/time_scale-min_rt)/(max_rt-min_rt) # second -> minute
                # fill in X again:
                for aa_index, aa in enumerate(seq[::-1]):
                    X[seq_index+N_seq, dictionary.word2idx[aa], aa_index] = 1                
                # fill in y again:
                self.y[seq_index+N_seq] = (seq_data.iloc[seq_index]['RT']/time_scale-min_rt)/(max_rt-min_rt) # second -> minute
            # store X:
            print('>> note: using sparse matrix to store the data.')
            self.X = [sparse.csr_matrix(i) for i in X]

        # for debugging:
        # self.X = self.X[:1500]
        # self.y = self.y[:1500]

        print('>> Read RT dataset done; source:', path)

#%% ============================== Language Model ==============================

class Corpus(object):
    '''
    DeepSA
    '''
    def __init__(self, dictionary, train_path, val_path='', test_path='', pad_length=0):
        '''
        test_path='': when this is blank, we only use train_path data for testing 
        pad_length: generally it's the max length of the seqs, but we can also specify this manually
        '''
        self.dictionary = dictionary

        # Add words to the dictionary, generally there is no new char in test data file, but we still do this again
        seq_data = pd.read_csv(train_path, sep = '\t')
        for aa in sorted(set(''.join(seq_data['sequence'].values))):
            self.dictionary.add_word(aa)

        # max length:
        if 0 == pad_length:
            self.max_length = max(seq_data['sequence'].str.len())
            print('DeepRT: using max length in training data:', self.max_length) # DeepRT
        else:
            self.max_length = pad_length
            print('DeepRT: using max length defined by user:', self.max_length) # DeepRT        

        # train data:
        self.train, self.train_label = self.tokenize(train_path, pad_length=0)
        print('Read training data done; source:', train_path)
        # self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        
        # val data:
        if '' != val_path:
            self.val, self.val_label = self.tokenize(val_path, pad_length=0)
            print('Read validation data done; source:', val_path)
        else:
            print('Note: didn\'t load val data.' )        

        if '' != test_path:
            self.test, self.test_label = self.tokenize(test_path, pad_length=0)
            print('Read testing data done; source:', test_path)
        else:
            print('Note: didn\'t load test data.' )

        # print out the dictionary to see:
        for aa in sorted(self.dictionary.word2idx.items(), key=operator.itemgetter(1)):
            print(aa[1],'->',aa[0])

    def tokenize(self, path, pad_length=0):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary, generally there is no new char in test data file, but we still do this again
        seq_data = pd.read_csv(path, sep = '\t')
        # for aa in sorted(set(''.join(seq_data['sequence'].values))):
        #     self.dictionary.add_word(aa)

        
        ids = np.zeros((len(seq_data['sequence']), self.max_length), dtype=int) # Note: dtype
        # label = np.zeros((len(seq_data['sequence']), max_length)) # the padding value is 0 here, and we just do this as is
        label = np.zeros((len(seq_data['sequence']), 1)) # DeepRT
        '''
        num_data = len(seq_data['sequence'])
        ids = np.zeros((num_data*2, self.max_length), dtype=int) # Note: dtype
        # label = np.zeros((len(seq_data['sequence']), max_length)) # the padding value is 0 here, and we just do this as is
        label = np.zeros((num_data*2, 1)) # DeepRT
        '''

        # Tokenize file content
        for index,seq in enumerate(seq_data['sequence'].values):
            ids[index, -len(seq):] = [self.dictionary.word2idx[aa] for aa in seq] # pad it at the front
            # ids[index+num_data, -len(seq):] = [self.dictionary.word2idx[aa] for aa in seq[::-1]] # data augmentation

        for index,obse in enumerate(seq_data['RT'].values):
            # label[index, -len(obse):] = [float(value) for value in obse.split(';')]
            label[index, 0] = (float(obse)/time_scale-min_rt)/(max_rt-min_rt) # float(obse) / 60. # DeepRT
            # label[index+num_data, 0] = float(obse) / 60. # DeepRT # data augmentation

        ids = torch.LongTensor(ids) # Note: the char index to be embedded has to be int!
        label = torch.FloatTensor(label)

        ids = ids.contiguous()
        label = label.contiguous()

        cuda = False
        if cuda:
            ids = ids.cuda()
            label = label.cuda()

        return ids, label

#%% ============================== Metrics ==============================
from math import sqrt
def RMSE(act, pred):
    '''
    accept two numpy arrays
    '''
    return sqrt(np.mean(np.square(act - pred)))

from scipy.stats import pearsonr
def Pearson(act, pred):
    return pearsonr(act, pred)[0]

from scipy.stats import spearmanr
def Spearman(act, pred):
    '''
    Note: there is no need to use spearman correlation for now
    '''
    return spearmanr(act, pred)[0]

def Delta_t95(act, pred):
    num95 = int(np.ceil(len(act) * 0.95))
    return 2 * sorted(abs(act - pred))[num95 - 1]

def Delta_tr95(act, pred):
    return Delta_t95(act, pred) / (max(act) - min(act))

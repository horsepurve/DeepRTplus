# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 03:54:59 2018

@author: horsepurve
"""

# generate the same data for Elude and GPTime
import sys

print('We then generate the same data for ELUDE and GPTime to use.')


train_file = sys.argv[1]
test_file = sys.argv[2]

digit2mod = dict({'4':'Y[80]',
                  '1':'M[16]', 
                  '3':'T[80]', 
                  '2':'S[80]' })
# digit2mod = None
# gptime - train: has K
fi = open(train_file, "r")
fo = open(train_file+'.gptime', "w")
seq_rts = fi.read().splitlines()
for seq_rt in seq_rts[1:]:
    seq = 'K.'+seq_rt.split('\t')[0]+'.K' # must have this, very weird
    if None != digit2mod:
        for mod in digit2mod.keys():
            seq = seq.replace(mod, digit2mod[mod])
    rt = str(float(seq_rt.split('\t')[1]) / 1.0)
    fo.write(seq+'\t'+rt)
    fo.write('\n')
fi.close()
fo.close()

# elude - train: no K
fi = open(train_file, "r")
fo = open(train_file+'.elude', "w")
seq_rts = fi.read().splitlines()
for seq_rt in seq_rts[1:]:
    seq = seq_rt.split('\t')[0] # must have this, very weird
    if None != digit2mod:
        for mod in digit2mod.keys():
            seq = seq.replace(mod, digit2mod[mod])
    rt = str(float(seq_rt.split('\t')[1]) / 1.0)
    fo.write(seq+'\t'+rt)
    fo.write('\n')
fi.close()
fo.close()

# gptime - test: has K, two columns
fi = open(test_file, "r")
fo = open(test_file+'.gptime', "w")
seq_rts = fi.read().splitlines()
for seq_rt in seq_rts[1:]:
    seq = 'K.'+seq_rt.split('\t')[0]+'.K'
    if None != digit2mod:
        for mod in digit2mod.keys():
            seq = seq.replace(mod, digit2mod[mod])
    rt = str(float(seq_rt.split('\t')[1]) / 1.0)
    fo.write(seq+'\t'+rt)
    fo.write('\n')
fi.close()
fo.close()

# elude - test: no K, one columns
fi = open(test_file, "r")
fo = open(test_file+'.elude', "w")
seq_rts = fi.read().splitlines()
for seq_rt in seq_rts[1:]:
    seq = seq_rt.split('\t')[0]
    if None != digit2mod:
        for mod in digit2mod.keys():
            seq = seq.replace(mod, digit2mod[mod])
    # rt = str(float(seq_rt.split('\t')[1]) / 60.0)
    fo.write(seq)
    fo.write('\n')
fi.close()
fo.close()

print('Data adaption done!')

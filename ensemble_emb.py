import torch
import numpy as np
from torch.autograd import Variable
from capsule_network_emb import *
import pickle
from sys import argv

def pred_from_model(conv1_kernel,
                    conv2_kernel,
                    param_path,                    
                    RTdata,
                    PRED_BATCH):
    '''
    write extracted features as np.array to pkl
    '''  
    model = CapsuleNet(conv1_kernel,conv2_kernel)
    model.load_state_dict(torch.load(param_path))
    model.cuda()
    
    print('>> note: predicting using the model:',param_path)
    
    pred = np.array([])
    
    # TODO: handle int
    pred_batch_number = int(RTdata.test.shape[0] / PRED_BATCH)+1
    for bi in range(pred_batch_number):
        test_batch = Variable(RTdata.test[bi*PRED_BATCH:(bi+1)*PRED_BATCH,:])
        test_batch = test_batch.cuda()
        pred_batch = model(test_batch)
        pred = np.append(pred, pred_batch[0].data.cpu().numpy().flatten())        
    return RTdata.test_label.numpy().flatten(), pred

import copy
def ensemble(obse,pred_list):
    pred_ensemble = copy.deepcopy(pred_list[0])
    for i in range(len(pred_list)-1):
        pred_ensemble += pred_list[i+1]
    pred_ensemble = pred_ensemble/len(pred_list)
    print('[ensemble %d] %.5f %.5f' %(len(pred_list),Pearson(obse,pred_ensemble),Delta_t95(obse,pred_ensemble)))
    return pred_ensemble    

def ensemble1round(job_seed_round,conv1,conv2,minrt,maxrt):
    batch = 100
    obse,pred1=pred_from_model(conv1,conv2,job_seed_round+'epoch_10.pt',RTtest,batch)
    _,pred2=pred_from_model(conv1,conv2,job_seed_round+'epoch_12.pt',RTtest,batch)
    _,pred3=pred_from_model(conv1,conv2,job_seed_round+'epoch_14.pt',RTtest,batch)
    _,pred4=pred_from_model(conv1,conv2,job_seed_round+'epoch_16.pt',RTtest,batch)
    _,pred5=pred_from_model(conv1,conv2,job_seed_round+'epoch_18.pt',RTtest,batch)
    S=maxrt-minrt
    obse,pred1,pred2,pred3,pred4,pred5=obse*S+minrt,pred1*S+minrt,pred2*S+minrt,pred3*S+minrt,pred4*S+minrt,pred5*S+minrt
    pred_ensemble=ensemble(obse,[pred1,pred2,pred3,pred4,pred5])
    return obse, pred_ensemble

# RTtest = RTdata(dictionary, max_length, test_path)
# desparse(RTtest)
corpus = Corpus(dictionary, # format: Corpus(dictionary, train_path, val_path='', test_path='', pad_length=0)
                train_path,
                test_path=test_path,
                pad_length=max_length)       
RTtest = corpus

# print(argv)
minrt = int(argv[1])
round1dir = argv[2] # 'work/dia/59/1/'
conv1 = int(argv[3])
round2dir = argv[4] # 'work/dia/59/2/'
conv2 = int(argv[5])
round3dir = argv[6] # 'work/dia/59/3/'
conv3 = int(argv[7])
result_ensemble = argv[8]
maxrt = int(argv[9])

obse, pred_r1 = ensemble1round(round1dir,conv1,conv1,minrt,maxrt)
_, pred_r2 = ensemble1round(round2dir,conv2,conv2,minrt,maxrt)
_, pred_r3 = ensemble1round(round3dir,conv3,conv3,minrt,maxrt)
pred_ensemble = ensemble(obse,[pred_r1,pred_r2,pred_r3])

# pred_ensemble = pred_r1
with open(result_ensemble, 'w') as fo:
    fo.write('observed\tpredicted\n')
    for i in range(len(obse)):
        fo.write('%.5f\t%.5f\n' % (obse[i],pred_ensemble[i]))

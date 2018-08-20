'''
We will not perform the data splitting process while running both resnet and lstm training. 
Rather, we will do it in the first place before everything.
'''
import sys
import os
import pandas as pd
import random

total_data = sys.argv[1]
train_ratio = int(sys.argv[2])
test_ratio = int(sys.argv[3])
random_seed = int(sys.argv[4])

def split_data(data_path, train_ratio, test_ratio, seed):
    '''
    generate the training and testing data using original data and random seed
    '''
    data = pd.read_csv(data_path).iloc[:,0].values.tolist()
    random.seed(seed)
    random.shuffle(data)
    cut = int(len(data)*train_ratio/(train_ratio+test_ratio))
    with open(data_path[:-4]+'_train_'+str(seed)+'.txt', 'w') as f:
        f.write('sequence\tRT\n')
        f.write('\n'.join(data[:cut])+'\n')
    with open(data_path[:-4]+'_test_'+str(seed)+'.txt', 'w') as f:
        f.write('sequence\tRT\n')
        f.write('\n'.join(data[cut:])+'\n')

print('Now split the data into training and testing data sets with the seed:'+str(random_seed)+'\n')
split_data(total_data, train_ratio, test_ratio, random_seed)
print('Done.')

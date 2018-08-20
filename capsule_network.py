"""
Dynamic Routing Between Capsules
https://arxiv.org/abs/1710.09829

PyTorch implementation by Kenta Iwasaki @ Gram.AI.
"""

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.autograd import Variable
import pickle
from scipy import sparse
from config import *

BATCH_SIZE = 16
NUM_CLASSES = 10
NUM_EPOCHS = 20
NUM_ROUTING_ITERATIONS = 1
CUDA = True
LR = 0.01

# train_path = 'data/dia_train_269.txt' # 'data/mod_train_2.txt' # 'data/unmod_train_2.txt' # 'data/SCX_train_42.txt' # 
# test_path = 'data/dia_test_269.txt' # 'data/mod_test_2.txt' # 'data/unmod_test_2.txt' # 'data/SCX_test_42.txt' # 
# result_path = 'dia_pred_269.txt' # 'mod_pred_2.txt' # 'unmod_pred_2.txt' # 'SCX_pred_42.txt' # 
RTdata_path = 'dia.pkl' # 'mod.pkl' # 'unmod.pkl' # 'SCX.pkl' # 
LOAD_DATA = True # False # 
## max_length = 66 # 50 # 38 # 

# log_path = ''
if '' == dict_path:
    dict_path = train_path

from RTdata import Dictionary, RTdata, Pearson, Spearman, Delta_t95, DATA_AUGMENTATION
dictionary = Dictionary(dict_path)
'''
if True == LOAD_DATA:
    dictionary = Dictionary(dict_path)
    RTtrain = RTdata(dictionary, max_length, train_path)
    RTtest = RTdata(dictionary, max_length, test_path)
    with open(RTdata_path, 'wb') as output:
        pickle.dump(dictionary, output)
        pickle.dump(RTtrain, output)
        pickle.dump(RTtest, output)
if False == LOAD_DATA:
    with open(RTdata_path, 'rb') as input:
        dictionary = pickle.load(input)
        RTtrain = pickle.load(input)
        RTtest = pickle.load(input)
        print('>> note: load pre-read RTdata from:', RTdata_path)

# DATA_AUGMENTATION = True
SPARSE = True
def desparse(RTtt):
    X = np.zeros((RTtt.number_seq, RTtt.N_aa, RTtt.N_time_step)) # DATA_AUGMENTATION->*2
    for i in range(RTtt.number_seq): # DATA_AUGMENTATION->*2
        # sparse to dense
        X[i,::] = RTtt.X[i].todense()
    RTtt.X = X
if True == SPARSE:
    print('>> note: de-sparse for both train & test data.')
    desparse(RTtrain)
    desparse(RTtest)
'''

def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    # print(transposed_input.contiguous().view(-1, transposed_input.size(-1)).shape)
    '''
    PyTorch 0.3.0:
    UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.
    '''
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)),dim=1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


def augmentation(x, max_shift=2):
    _, _, height, width = x.size()

    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = torch.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
    return shifted_image.float() # Note float here!


class CapsuleLayer(nn.Module):
    def __init__(self, 
                 num_capsules, 
                 num_route_nodes, 
                 in_channels, 
                 out_channels, 
                 kernel_size=None, 
                 stride=None,
                 num_iterations=NUM_ROUTING_ITERATIONS):

        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, 
                                                          num_route_nodes, 
                                                          in_channels, 
                                                          out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, 
                           out_channels, 
                           kernel_size=kernel_size, 
                           stride=stride, 
                           padding=0) for _ in
                 range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

            if True == CUDA:
                logits = Variable(torch.zeros(*priors.size())).cuda()
            if False == CUDA:
                logits = Variable(torch.zeros(*priors.size()))

            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs

param_2D = {'data' : 'mnist',
            'dim' : 2,
            'conv1_kernel' : 9,
            'pri_caps_kernel' : 9,
            'stride' : 2,
            'digit_caps_nodes' : 32 * 6 * 6,
            'NUM_CLASSES' : NUM_CLASSES}

param_1D = {'data' : 'mnist',
            'dim' : 1,
            'conv1_kernel' : (28, 9),
            'pri_caps_kernel' : (1, 9),
            'stride' : 1,
            'digit_caps_nodes' : 32 * 1 * 12,
            'NUM_CLASSES' : 1}

# conv1_kernel = 15
# conv2_kernel = 15
param_1D_rt = {'data' : 'rt',
               'dim' : 1,
               'conv1_kernel' : (len(dictionary), conv1_kernel),
               'pri_caps_kernel' : (1, conv2_kernel),
               'stride' : 1,
               'digit_caps_nodes' : 32 * 1 * (max_length - conv1_kernel*2 + 2 - conv2_kernel + 1), # 32 # Note: number of conv!
               'NUM_CLASSES' : 1}

param = param_1D_rt

if 2 == param['dim']:
    print('>> note: using image mode.')
if 1 == param['dim']:
    print('>> note: using seq mode.')

class CapsuleNet(nn.Module):
    def __init__(self,conv1_kernel,conv2_kernel):
        super(CapsuleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, 
                               out_channels=256, # 256
                               kernel_size=(len(dictionary), conv1_kernel), # param['conv1_kernel'], # (28, 9), # 9, 
                               stride=1)
        ''''''
        self.bn1 = nn.BatchNorm2d(256) # Note: do we need this or not?
        self.conv2 = nn.Conv2d(in_channels=256, 
                               out_channels=256, # 256
                               kernel_size=(1, conv1_kernel), # (28, 9), # 9, 
                               stride=1)
        self.bn2 = nn.BatchNorm2d(256)
        '''
        self.conv3 = nn.Conv2d(in_channels=128, 
                               out_channels=256, # 256
                               kernel_size=(1, conv1_kernel), # (28, 9), # 9, 
                               stride=1)
        self.bn3 = nn.BatchNorm2d(256)
        '''

        self.primary_capsules = CapsuleLayer(num_capsules=8, # 8
                                             num_route_nodes=-1, 
                                             in_channels=256, # 256
                                             out_channels=32, # 32
                                             kernel_size=(1, conv2_kernel), # param['pri_caps_kernel'], # (1, 9), # 9, 
                                             stride=param['stride']) # 1) # 2)

        self.digit_capsules = CapsuleLayer(num_capsules=param['NUM_CLASSES'], # 1, #NUM_CLASSES, # DeepRT
                                           num_route_nodes=32 * 1 * (max_length - conv1_kernel*2 + 2 - conv2_kernel + 1), # param['digit_caps_nodes'], # 32 * 1 * 12, # 32 * 6 * 6, 
                                           in_channels=8, # 8
                                           out_channels=16) # max_length-conv1_kernel + 1) # 16

        # add dropout:
        # self.dropout = nn.Dropout(0.1) # not good!
        # self.linear = nn.Linear((max_length-conv1_kernel+1)*256,16) # try residue: not good!
        ''' residue is not very good!
        pad = 0
        kernel_h = pad*2+1 # len + pad*2 - (kernel_h - 1) = len
        self.conv_res = nn.Conv2d(in_channels=256, 
                                  out_channels=1, # 256
                                  kernel_size=(1, kernel_h), # (28, 9), # 9, 
                                  stride=1)
                                  #padding =(0,pad))
        '''
        self.decoder = nn.Sequential(
            nn.Linear(16 * NUM_CLASSES, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        # print('>>dim: input', x.shape) # [batch, 1, 28, 28]
        # print('>>dim: y', y) # [batch, 10] ~ [batch, NUM_CLASSES]
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)      

        ''' try residue: not good! 
        residue = x.view(x.shape[0],-1)        
        residue = self.linear(residue).view(residue.shape[0],1,16)
        # another residue method
        residue = F.relu(self.conv_res(x), inplace=True)
        residue = residue.view(residue.shape[0],1,residue.shape[-1])
        '''

        # x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True) # improvement
        # x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        # print('>>dim: conv1', x.shape) # [batch, 256, 20, 20]
        x = self.primary_capsules(x)
        # print('>>dim: primary_capsules', x.shape) # [batch, 1152, 8] = [batch, 6*6*32, 8]
        # print('>>dim: unsqueezeed', self.digit_capsules(x).shape) # [10, batch, 1, 1, 16] ~ [num_caps, batch, ...]
        if 2 == param['dim']:
            x = self.digit_capsules(x).squeeze().transpose(0, 1) # DeepRT
            # [10, batch, 1, 1, 16] -> squeeze: [10, batch, 16] -> transpose: [batch, 10, 16]
        if 1 == param['dim']:
            x = self.digit_capsules(x).squeeze()[:, None, :]
            # [1, batch, 1, 1, 16] -> squeeze: [batch, 16]
        # print('>>dim: digit_capsules', x.shape) # [batch, 10, 16]

        # add dropout:
        # x = self.dropout(x)
        # x = self.linear(x)
        # x = F.sigmoid(x)
        
        # x = x + residue # try residue: not good!
        classes = (x ** 2).sum(dim=-1) ** 0.5
        # print('>>dim: classes', classes) # [batch, 10]
        if 2 == param['dim']:
            classes = F.softmax(classes) # DeepRT
        # print('>>dim: softmax', classes)

        if y is None: # Note: not do this during training. Here y is only used for reconstruction
            if 2 == param['dim']:
                # In all batches, get the most active capsule.
                # print('>>dim: reconstruction', classes) # [batch, 10]
                _, max_length_indices = classes.max(dim=1) 
                # give: [torch.FloatTensor of size batch] and [torch.FloatTensor of size batch]
                if True == CUDA:
                    y = Variable(torch.sparse.torch.eye(NUM_CLASSES)).cuda().index_select(dim=0, index=max_length_indices.data)
                if False == CUDA:
                    y = Variable(torch.sparse.torch.eye(NUM_CLASSES)).index_select(dim=0, index=max_length_indices.data)
                # generate a new y: [batch, 10] with each column having 1 in batch 0

        if 2 == param['dim']:
            # print('>>dim: x*y', x.shape, y.shape)
            reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
           # x: [batch, 10, 16], y: [batch, 10] -> [batch, 10, 1]
            return classes, reconstructions
        if 1 == param['dim']:
            return classes, x # Note here

class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, images, labels, classes, reconstructions):
        if 2 == param['dim']:
            # print('>>dim: labels', labels) # [batch, 10]
            # print('>>dim: classes', classes) # [batch, 10]
            left = F.relu(0.9 - classes, inplace=True) ** 2
            right = F.relu(classes - 0.1, inplace=True) ** 2

            margin_loss = labels * left + 0.5 * (1. - labels) * right
            margin_loss = margin_loss.sum()

            reconstruction_loss = self.reconstruction_loss(reconstructions, images)

            loss = (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)
            # print('>>dim: loss', loss) # it's a single value
            return loss
        if 1 == param['dim']:
            # print('>>dim: labels', labels) # torch.cuda.FloatTensor of size batch x 1
            # print('>>dim: classes', classes) # [batch, 1]

            '''
            square = (labels - classes) ** 2
            square = square.sort(dim=0,descending=False)[0]
            cut = int(labels.shape[0]-1)
            loss = (square[:cut]).sum()/cut
            loss = loss ** 0.5 + square[cut] ** 0.5
            '''
            loss = ((labels - classes) ** 2).sum()/labels.shape[0] # MSE # Note: here it must be sum()
            loss = loss ** 0.5 # RMSE
            
            # print('>>dim: loss', loss)
            return loss

def desparse(RTtt):
    if False == DATA_AUGMENTATION:
        X = np.zeros((RTtt.number_seq, RTtt.N_aa, RTtt.N_time_step)) # DATA_AUGMENTATION->*2
        for i in range(RTtt.number_seq): # DATA_AUGMENTATION->*2
            # sparse to dense
            X[i,::] = RTtt.X[i].todense()
        RTtt.X = X
    else:
        print('>> note: usnig data_augmentation')
        X = np.zeros((RTtt.number_seq*2, RTtt.N_aa, RTtt.N_time_step)) # DATA_AUGMENTATION->*2
        for i in range(RTtt.number_seq*2): # DATA_AUGMENTATION->*2
            # sparse to dense
            X[i,::] = RTtt.X[i].todense()
        RTtt.X = X

if __name__ == "__main__":
    # from torch.autograd import Variable
    from torch.optim import Adam
    from torchnet.engine import Engine
    # from torchnet.logger import VisdomPlotLogger, VisdomLogger
    # from torchvision.utils import make_grid
    # from torchvision.datasets.mnist import MNIST
    from tqdm import tqdm
    import torchnet as tnt
    import gc 
    from time import sleep, time
    import timeit
    T1 = timeit.default_timer()

    # read data ========== ========== ========== ========== ========== ==========
    if True == LOAD_DATA:
        # dictionary = Dictionary(dict_path)
        RTtrain = RTdata(dictionary, max_length, train_path)
        RTtest = RTdata(dictionary, max_length, test_path)
        with open(RTdata_path, 'wb') as output:
            # pickle.dump(dictionary, output)
            pickle.dump(RTtrain, output)
            pickle.dump(RTtest, output)
    if False == LOAD_DATA:
        with open(RTdata_path, 'rb') as input:
            # dictionary = pickle.load(input)
            RTtrain = pickle.load(input)
            RTtest = pickle.load(input)
            print('>> note: load pre-read RTdata from:', RTdata_path)

    # DATA_AUGMENTATION = True
    SPARSE = True
    # def desparse(RTtt):
    #     X = np.zeros((RTtt.number_seq, RTtt.N_aa, RTtt.N_time_step)) # DATA_AUGMENTATION->*2
    #     for i in range(RTtt.number_seq): # DATA_AUGMENTATION->*2
    #         # sparse to dense
    #         X[i,::] = RTtt.X[i].todense()
    #     RTtt.X = X
    if True == SPARSE:
        print('>> note: de-sparse for both train & test data.')
        desparse(RTtrain)
        desparse(RTtest)
    # read data ========== ========== ========== ========== ========== ==========

    LOG = False
    flog = open(log_path, 'w')

    model = CapsuleNet(conv1_kernel,conv2_kernel)
    if '' == pretrain_path:
        pass
    else:
        model.load_state_dict(torch.load(pretrain_path)) # epoch.pt
        print('>> note: load pre-trained model from:',pretrain_path)

    if True == CUDA:
        model.cuda()

    print("# parameters:", sum(param.numel() for param in model.parameters()))
    flog.write("# parameters:"+str(sum(param.numel() for param in model.parameters()))+'\n')

    optimizer = Adam(model.parameters(), lr = LR)

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    if 2 == param['dim']:        
        meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
        confusion_meter = tnt.meter.ConfusionMeter(NUM_CLASSES, normalized=True)
    if 1 == param['dim']:
        pass
        # meter_mse = tnt.meter.MSEMeter()

    if True == LOG:
        train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'})
        train_error_logger = VisdomPlotLogger('line', opts={'title': 'Train Accuracy'})
        test_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Loss'})
        test_accuracy_logger = VisdomPlotLogger('line', opts={'title': 'Test Accuracy'})
        confusion_logger = VisdomLogger('heatmap', opts={'title': 'Confusion matrix',
                                                     'columnnames': list(range(NUM_CLASSES)),
                                                     'rownames': list(range(NUM_CLASSES))})
        if 2 == param['dim']:
            ground_truth_logger = VisdomLogger('image', opts={'title': 'Ground Truth'})
            reconstruction_logger = VisdomLogger('image', opts={'title': 'Reconstruction'})

    capsule_loss = CapsuleLoss()

    def get_iterator(mode):
        dataset = MNIST(root='./data', download=True, train=mode)
        data = getattr(dataset, 'train_data' if mode else 'test_data')[:47]
        # [torch.ByteTensor of size number x 28 x 28]
        labels = getattr(dataset, 'train_labels' if mode else 'test_labels')[:47]
        # [torch.LongTensor of size number]
        tensor_dataset = tnt.dataset.TensorDataset([data, labels])

        return tensor_dataset.parallel(batch_size=BATCH_SIZE, num_workers=4, shuffle=mode)

    data_train = torch.FloatTensor(RTtrain.X)
    label_train = torch.FloatTensor(RTtrain.y)
    print('>> note: delete RTtrain.')
    del RTtrain
    gc.collect()
    print('>> sleeping...')
    for i in range(5):
        print('~.~')
    print('>> wake up!')   
     
    def get_rt_iterator(mode):
        if mode:
            data = data_train # Note: here must be FloatTensor not ByteTensor!            
            labels = label_train
        else:
            data = torch.FloatTensor(RTtest.X)
            labels = torch.FloatTensor(RTtest.y)
            # print('>>dim: test data:', data.shape, labels.shape)
        tensor_dataset = tnt.dataset.TensorDataset([data, labels])
        return tensor_dataset.parallel(batch_size=BATCH_SIZE, num_workers=4, shuffle=mode) # 1 for heatmap

    def processor(sample):
        data, labels, training = sample
        # print('>>dim: data, labels, training', data.shape, labels.shape, training)
        # torch.Size([batch, 28, 28]) torch.Size([batch]) True

        if 'mnist' == param['data']:
            data = augmentation(data.unsqueeze(1).float() / 255.0)            
        # print('>>dim: data augmentation', data.shape) # torch.Size([batch, 1, 28, 28])
        # print('>>dim: labels', labels) # Note: labels is already LongTensor?
        if 'rt' == param['data']:
            data = data[:, None, :, :]            

        if 2 == param['dim']:
            # for classification, we use LongTensor
            labels = torch.LongTensor(labels)
            labels = torch.sparse.torch.eye(NUM_CLASSES).index_select(dim=0, index=labels) 
        if 1 == param['dim']:
            # for regression, we use FloatTensor
            labels = torch.FloatTensor(labels.numpy())
            labels = labels.view(-1, 1) # from [batch] to [batch, 1]

        if True == CUDA:
            data = Variable(data).cuda()
            labels = Variable(labels).cuda()
        if False == CUDA:
            data = Variable(data)
            labels = Variable(labels)

        if training:
            classes, reconstructions = model(data, labels)
        else:
            classes, reconstructions = model(data)

        loss = capsule_loss(data, labels, classes, reconstructions)

        return loss, classes


    def reset_meters():
        meter_loss.reset()
        if 2 == param['dim']:
            meter_accuracy.reset()            
            confusion_meter.reset()
        if 1 == param['dim']:
            pass
            # meter_mse.reset()

    def on_sample(state):
        state['sample'].append(state['train'])


    def on_forward(state):
        '''
        So it is just used for recording?
        '''
        if 1 == param['dim']:
            # print('>>dim: state output', state['output'].data.view(-1)) 
            # torch.FloatTensor of size [batch x 10]
            # print('>>dim: state sample', state['sample'][1]) 
            # torch.LongTensor of size [batch]
            # (1): [batch, 1] (2): [batch], so we view (1) as [batch], but no view is fine     
            pass       
            # meter_mse.add(state['output'].data, torch.FloatTensor(state['sample'][1].numpy()))
        if 2 == param['dim']:
            meter_accuracy.add(state['output'].data, torch.LongTensor(state['sample'][1]))
            confusion_meter.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        meter_loss.add(state['loss'].data[0])


    def on_start_epoch(state):
        reset_meters()
        state['iterator'] = tqdm(state['iterator'])


    def on_end_epoch(state):
        if 2 == param['dim']:
            print('[Epoch %d] Training Loss: %.4f (Accuracy: %.2f%%)' % (
                state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))
            flog.write('[Epoch %d] Training Loss: %.4f (Accuracy: %.2f%%)\n' % (
                state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))
            if True == LOG:
                train_loss_logger.log(state['epoch'], meter_loss.value()[0])
                train_error_logger.log(state['epoch'], meter_accuracy.value()[0])
        if 1 == param['dim']:
            print('[Epoch %d] Training Loss: %.4f (MSE: %.4f)' % (
                state['epoch'], meter_loss.value()[0], 7)) # meter_mse.value()
            flog.write('[Epoch %d] Training Loss: %.4f (MSE: %.4f)\n' % (
                state['epoch'], meter_loss.value()[0], 7)) # meter_mse.value()

        reset_meters()

        # iterator
        if 'mnist' == param['data']:
            engine.test(processor, get_iterator(False))
        if 'rt' == param['data']:
            engine.test(processor, get_rt_iterator(False))

        if True == LOG:
            test_loss_logger.log(state['epoch'], meter_loss.value()[0])
            if 2 == param['dim']:
                test_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])
                confusion_logger.log(confusion_meter.value())
            if 1 == param['dim']:
                test_accuracy_logger.log(state['epoch'], 7) # meter_mse.value()

        if 2 == param['dim']:
            print('[Epoch %d] Testing Loss: %.4f (Accuracy: %.2f%%)' % (
                state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))
            flog.write('[Epoch %d] Testing Loss: %.4f (Accuracy: %.2f%%)\n' % (
                state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))
        if 1 == param['dim']:
            print('[Epoch %d] Testing Loss: %.4f (MSE: %.4f)' % (
                state['epoch'], meter_loss.value()[0], 7)) # meter_mse.value()
            flog.write('[Epoch %d] Testing Loss: %.4f (MSE: %.4f)\n' % (
                state['epoch'], meter_loss.value()[0], 7)) # meter_mse.value()

        if 10 <= state['epoch']: # for heatmap
            torch.save(model.state_dict(), save_prefix+'/epoch_%d.pt' % state['epoch'])
            print('>> model: saved.')        

        # prediction:
        # model.load_state_dict(torch.load(PATH))
        # pred_data = Variable(torch.FloatTensor(RTtest.X)[:,None,:,:])        
        PRED_BATCH = 16 # 1000 # 16 for heatmap

        if PRED_BATCH > 0:
            '''
            solve memory problem using batch
            '''
            pred = np.array([])
            # TODO: handle int
            pred_batch_number = int(RTtest.X.shape[0] / PRED_BATCH)+1
            for bi in range(pred_batch_number):
                test_batch = Variable(torch.FloatTensor(RTtest.X[bi*PRED_BATCH:(bi+1)*PRED_BATCH,:,:])[:,None,:,:])
                test_batch = test_batch.cuda()
                pred_batch = model(test_batch)
                pred = np.append(pred, pred_batch[0].data.cpu().numpy().flatten())
            # print('>>dim: pred', pred.shape)      

            if True == DATA_AUGMENTATION:
                ''' data augmentation:'''
                pep_num = int(len(pred) / 2)
                pred = pred[:pep_num]*0.5 + pred[pep_num:]*0.5
                obse = RTtest.y[:pep_num]
                pearson = Pearson(pred,obse)
                spearman = Spearman(pred,obse)
            else:
                pearson = Pearson(pred,RTtest.y)
                spearman = Spearman(pred,RTtest.y)            
        
        else:
            pred_data = Variable(torch.FloatTensor(RTtest.X)[:,None,:,:]) 
            if True == CUDA:
                pred_data = pred_data.cuda()
            pred = model(pred_data)
            if True == CUDA:
                # print('>>dim: pred', pred[0].data.cpu().numpy().flatten().shape)
                pearson = Pearson(pred[0].data.cpu().numpy().flatten(),RTtest.y)
                spearman = Spearman(pred[0].data.cpu().numpy().flatten(),RTtest.y)
            if False == CUDA:
                pearson = Pearson(pred[0].data.numpy().flatten(),RTtest.y)
                spearman = Spearman(pred[0].data.numpy().flatten(),RTtest.y)
        ''''''
        print('>> Corr on %d testing samples: %.5f | %.5f' % (len(pred), pearson, spearman))
        flog.write('>> Corr on %d testing samples: %.5f | %.5f\n' % (len(pred), pearson, spearman))
        # writing:
        with open(result_path, 'w') as fo:
                fo.write('observed\tpredicted\n')
                for i in range(len(pred)):
                    fo.write('%.5f\t%.5f\n' % (RTtest.y[i],pred[i]))
        # writing done

        # Reconstruction visualization.
        if 2 == param['dim']:

            # iterator
            if 'mnist' == param['data']:
                test_sample = next(iter(get_iterator(False)))
            if 'rt' == param['data']:
                test_sample = next(iter(get_rt_iterator(False)))
            # print('>>dim: test_sample', test_sample) # [batch, 28, 28]

            ground_truth = (test_sample[0].unsqueeze(1).float() / 255.0)
            # print('>>dim: ground_truth', ground_truth.shape) # torch.FloatTensor of size batch x 1 x 28 x 28

            if True == CUDA:
                pred, reconstructions = model(Variable(ground_truth).cuda())
            if False == CUDA:
                pred, reconstructions = model(Variable(ground_truth))
            # print('>>dim: pred', pred)
            reconstruction = reconstructions.cpu().view_as(ground_truth).data

            if True == LOG:
                ground_truth_logger.log(
                    make_grid(ground_truth, nrow=int(BATCH_SIZE ** 0.5), normalize=True, range=(0, 1)).numpy())
                reconstruction_logger.log(
                    make_grid(reconstruction, nrow=int(BATCH_SIZE ** 0.5), normalize=True, range=(0, 1)).numpy())

    # def on_start(state):
    #     state['epoch'] = 327
    #
    # engine.hooks['on_start'] = on_start

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    if 'mnist' == param['data']:
        engine.train(processor, get_iterator(True), maxepoch=NUM_EPOCHS, optimizer=optimizer)
    if 'rt' == param['data']:
        engine.train(processor, get_rt_iterator(True), maxepoch=NUM_EPOCHS, optimizer=optimizer)

    T2 = timeit.default_timer()
    print('>> time: %.5f min\n' %((T2-T1)/60.))
    flog.write('>> time: %.5f min\n' %((T2-T1)/60.))
    flog.close()

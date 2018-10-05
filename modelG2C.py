# this script is used to convert GPU model to CPU model

gpu_model_path = 'param/dia_all_epo20_dim24_conv10/dia_all_epo20_dim24_conv10_filled.pt'
cpu_model_path = 'param_cpu/dia_all_epo20_dim24_conv10/dia_all_epo20_dim24_conv10_filled.pt'

from capsule_network_emb import *

model = CapsuleNet(conv1_kernel,conv2_kernel)

model.load_state_dict(torch.load(gpu_model_path)) # epoch.pt
print('>> note: load pre-trained model from:',pretrain_path)

model.cpu()

torch.save(model.state_dict(), cpu_model_path)
print('>> model: saved.')   
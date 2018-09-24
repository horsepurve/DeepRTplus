# This script shows how to install prerequisites for DeepRT.
# first, install pytorch via https://pytorch.org, the version used here is pytorch 0.3.0 with CUDA 9.1 and Python 3.6. PyTorch can be run on both Windows (win10) and Linux (Ubuntu, CentOS).
# install torchvision as well.

# install the following python packages using conda or pip:
# numpy
# scipy
# matplotlib
# pickle

# install TorchNet via https://github.com/torchnet/torchnet:
pip install torchnet

# install tnt via https://github.com/pytorch/tnt:
pip install git+https://github.com/pytorch/tnt.git@master

# install TQDM, e.g.:
pip install tqdm

# install Visdom via https://github.com/facebookresearch/visdom, like:
pip install visdom

# All done!
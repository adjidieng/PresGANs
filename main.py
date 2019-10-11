from __future__ import print_function

import argparse
import os
import random
import numpy as np 
import pickle 
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
import torch.nn.functional as F 

import utils 
import data 
import nets
import train

parser = argparse.ArgumentParser()

###### Data arguments
parser.add_argument('--dataset', required=True, help=' ring | mnist | stackedmnist | cifar10 ')
parser.add_argument('--dataroot', type=str, default='data', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2) 
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--Ntrain', type=int, default=60000, help='training set size for stackedmnist')
parser.add_argument('--Ntest', type=int, default=10000, help='test set size for stackedmnist')

###### Model arguments
parser.add_argument('--model', required=True, help=' dcgan | presgan ')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)

###### Optimization arguments
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrE', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--seed', type=int, default=2019, help='manual seed')

###### Checkpointing and Logging arguments
parser.add_argument('--log_hmc', type=int, default=500, help='reporting')
parser.add_argument('--ckptG', type=str, default='', help='a given checkpoint file for generator')
parser.add_argument('--ckptD', type=str, default='', help='a given checkpoint file for discriminator')
parser.add_argument('--logsigma_file', type=str, default='', help='a given file for logsigma for the generator')
parser.add_argument('--log', type=int, default=200, help='when to log')
parser.add_argument('--save_ckpt_every', type=int, default=1, help='when to save checkpoint')
parser.add_argument('--save_imgs_every', type=int, default=1, help='when to save generated images')
parser.add_argument('--num_gen_images', type=int, default=150, help='number of images to generate for inspection')

###### PresGAN-specific arguments
parser.add_argument('--sigma_lr', type=float, default=0.0002, help='generator variance')
parser.add_argument('--lambda_', type=float, default=0.01, help='entropy coefficient')
parser.add_argument('--sigma_min', type=float, default=0.01, help='min value for sigma')
parser.add_argument('--sigma_max', type=float, default=0.3, help='max value for sigma')
parser.add_argument('--logsigma_init', type=float, default=-1.0, help='initial value for log_sigma_sian')
parser.add_argument('--num_samples_posterior', type=int, default=2, help='number of samples from posterior')
parser.add_argument('--burn_in', type=int, default=2, help='hmc burn in')
parser.add_argument('--leapfrog_steps', type=int, default=5, help='number of leap frog steps for hmc')
parser.add_argument('--flag_adapt', type=int, default=1, help='0 or 1')
parser.add_argument('--delta', type=float, default=1.0, help='delta for hmc')
parser.add_argument('--hmc_learning_rate', type=float, default=0.02, help='lr for hmc')
parser.add_argument('--hmc_opt_accept', type=float, default=0.67, help='hmc optimal acceptance rate')
parser.add_argument('--save_sigma_every', type=int, default=1, help='interval to save sigma for sigan traceplot')
parser.add_argument('--stepsize_num', type=float, default=1.0, help='initial value for hmc stepsize')
parser.add_argument('--restrict_sigma', type=int, default=0, help='whether to restrict sigma or not')


args = parser.parse_args()

if args.model == 'presgan':
    args.results_folder = args.model+'_lambda_'+str(args.lambda_)
else:
    args.results_folder = args.model

print('\nTraining with the following settings: {}'.format(args))

if not os.path.exists(args.results_folder):
    os.makedirs(args.results_folder)

if not os.path.exists(args.dataroot):
    os.makedirs(args.dataroot)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device 

np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)
cudnn.benchmark = True


dat = data.load_data(args.dataset, args.dataroot, args.batchSize, 
                        device=device, imgsize=args.imageSize, Ntrain=args.Ntrain, Ntest=args.Ntest)

#### defining generator
netG = nets.Generator(args.imageSize, args.nz, args.ngf, dat['nc']).to(device)
if args.model == 'presgan':
    log_sigma = torch.tensor([args.logsigma_init]*(args.imageSize*args.imageSize), device=device, requires_grad=True)
print('{} Generator: {}'.format(args.model.upper(), netG))

#### defining discriminator
netD = nets.Discriminator(args.imageSize, args.ndf, dat['nc']).to(device) 
print('{} Discriminator: {}'.format(args.model.upper(), netD))

#### initialize weights
netG.apply(utils.weights_init)
if args.ckptG != '':
    netG.load_state_dict(torch.load(args.ckptG))
netD.apply(utils.weights_init)
if args.ckptD != '':
    netD.load_state_dict(torch.load(args.ckptD))

#### train a given model
if args.model == 'dcgan':
    train.dcgan(dat, netG, netD, args)
elif args.model == 'presgan':
    train.presgan(dat, netG, netD, log_sigma, args)
else:
    raise NotImplementedError('Your model is not supported yet :-(')

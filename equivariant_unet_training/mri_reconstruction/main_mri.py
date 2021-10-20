
import argparse
import os
import random
import shutil
import time
import warnings
import sys
import numpy as np
import os
from collections import defaultdict 
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

import torchvision


from runstats import Statistics


import pickle

import sys

sys.path.insert(1, '../')
sys.path.insert(1, '../../')


import models
from models.baseline_models import baseline_unet
from models.aps_models import unet_aps
from models.lpf_models import unet_lpf


import fastmri
from fastmri.evaluate import psnr
from fastmri.data import transforms as fastmri_transforms

from utils.load_fastmri_dataset import get_dataloaders_fastmri
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

import h5py



parser = argparse.ArgumentParser(description='PyTorch FastMRI Training')

parser.add_argument('--out-dir', dest='out_dir', default='/mnt/ext6TB/anadi/shift_equivariant_nets_results/mri_exps_results/', type=str,
					help='output directory')
														
parser.add_argument('--data', metavar='DIR', default='/mnt/ext6TB/fastmri_latest_download/',
					help='path to dataset')
					  
#..................training and evaluation args............
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
					help='number of data loading workers (default: 4)')
parser.add_argument('-ep', '--epochs', default=50, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
					metavar='N',
					help='mini-batch size, this is the total '
						 'batch size of all GPUs on the current node when '
						 'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--train_sample_complexity_frac', default=1.0, type=float,
					help='Fraction of training samples to train on.')



parser.add_argument('--criterion', default='nn.MSELoss', type=str,
					help='Loss function to be optimized over. ')

parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
					metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_step', default=5, type=float,
					help='number of epochs before stepping down learning rate')

parser.add_argument('--lr_decay_factor', default=0.5, type=float,
					help='Multiplicative decay factor when stepping down lr.')


parser.add_argument('--cos_lr', action='store_true',
					help='use cosine learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0.0, type=float,
					metavar='W', help='weight decay (default: 1e-4)',
					dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
					metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
					help='use pre-trained model')
parser.add_argument('--force_nonfinetuned', dest='force_nonfinetuned', action='store_true',
					help='if pretrained, load the model that is pretrained from scratch (if available)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')

parser.add_argument('--world-size', default=-1, type=int,
					help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
					help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
					help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
					help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
					help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
					help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
					help='Use multi-processing distributed training to launch '
						 'N processes per node, which has N GPUs. This is the '
						 'fastest way to use PyTorch for either single node or '
						 'multi node data parallel training')


parser.add_argument('--visible_devices', default='all', type=str,
					help='GPU devices to be made visible in the beginning. Give device ids separated by commas')

parser.add_argument('--circular_data_aug', action='store_true',
					help='circular shift-based data augmentation')


parser.add_argument('-e_adv', '--evaluate_psnr_adverserial', dest='evaluate_psnr_adverserial', action='store_true',
					help='evaluate model on adverserial shifts.')


parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
					help='evaluate model on validation set for PSNR.')

parser.add_argument( '--evaluate-save-unshifted', dest='evaluate_save_unshifted', action='store_true',
					help='Save observed and reconstructed images ')

parser.add_argument('--eval_phase', dest='eval_phase', default = 'validation',
					help='Dataset split (train/val/test) to evaluate trained model.')


parser.add_argument('-es', '--evaluate-random-shift', dest='evaluate_random_shift', action='store_true',
					help='evaluate model for PSNR on randomly shifted validation images.')


parser.add_argument('--evaluate-equivariance-metrics', dest='evaluate_equivariance_metrics', action='store_true',
					help='Evaluate NMSE between Shift(f(x)) and f(shift(x)), with x denoting validation set images.')

parser.add_argument('--evaluate-psnr-diff-shifts', dest='evaluate_psnr_diff_random_shifts', action='store_true',
					help='Evaluate abs difference between psnr obtained with unshifted and randomly shifted images.')

parser.add_argument('--evaluate-vert-flip-psnr-diff-shifts', dest='evaluate_vert_flip_diff_psnr_random_shifts', action='store_true',
					help='Evaluate abs difference between psnr obtained with vertically flipped unshifted and randomly shifted images.')


parser.add_argument('--psnr_diff_eval_num_shifts', type = int, default = 1, 
	help = 'Number of passes through the dataset while evaluating decline in psnr caused by random shifts.')


parser.add_argument('--evaluate-out-of-dist-noise-equivariance', dest='evaluate_out_of_dist_noise_equivariance', action='store_true',
					help='Evaluate equivariance metrics on noisy val images.')

parser.add_argument('--evaluate-out-of-dist-ct-equivariance', dest='evaluate_out_of_dist_ct_equivariance', action='store_true',
					help='Evaluate equivariance metrics on test images from CT dataset.')

parser.add_argument('--evaluate-out-of-dist-wgn-equivariance', dest='evaluate_out_of_dist_wgn_equivariance', action='store_true',
					help='Evaluate equivariance metrics on test images from CT dataset.')

parser.add_argument('--evaluate-out-of-dist-imagenet-equivariance', dest='evaluate_out_of_dist_imagenet_equivariance', action='store_true',
					help='Evaluate equivariance metrics on first few images of imagenet val dataset.')


parser.add_argument('--wgn_out_of_dist_num', type = int, default = 1000, 
	help = 'Number of  wgn images to be tested on for out of dist equivariance')


parser.add_argument('--imagenet_out_of_dist_num', type = int, default = 1000, 
	help = 'Number of  wgn images to be tested on for out of dist equivariance')



parser.add_argument('--evaluate-out-of-dist-psnr-diff-with-shift', dest='evaluate_out_of_dist_psnr_diff_with_shift', action='store_true',
					help='Evaluate abs difference between psnr obtained with unshifted and randomly shifted noisy images.')


parser.add_argument('--evaluate-out-of-dist-noise-unshifted', dest='evaluate_out_of_dist_noise_unshifted_metrics', action='store_true',
					help='Evaluate performance metrics on unshifted noisy val images.')



parser.add_argument('--eval_noise_level', default = '0.0', type = float,
					help='Standard deviation of white Gaussian noise to be added in out-of-dist experiments.')

parser.add_argument('--evaluate_acquisition', default = None,
					help='Evaluate on images only with a specific fat suppression mode. Possible choices: None, PDFS and PD.')


parser.add_argument('--max_shift', default = 16, type = int,
					help='evaluate model for PSNR on shifted validation images.')


parser.add_argument('--evaluate-save', dest='evaluate_save', action='store_true',
					help='save validation images off')

parser.add_argument('--epochs-shift', default=1, type=int, metavar='N',
					help='number of total epochs to run for shift-invariance test')
parser.add_argument('-ed', '--evaluate-diagonal', dest='evaluate_diagonal', action='store_true',
					help='evaluate model on diagonal')



#.................  misc args ................................
parser.add_argument('-ba', '--batch-accum', default=1, type=int,
					metavar='N',
					help='number of mini-batches to accumulate gradient over before updating (default: 1)')
parser.add_argument('--embed', dest='embed', action='store_true',
					help='embed statement before anything is evaluated (for debugging)')
parser.add_argument('--val-debug', dest='val_debug', action='store_true',
					help='debug by training on val set')
parser.add_argument('--weights', default=None, type=str, metavar='PATH',
					help='path to pretrained model weights')
parser.add_argument('--save_weights', default=None, type=str, metavar='PATH',
					help='path to save model weights')
parser.add_argument('--finetune', action='store_true', help='finetune from baseline model')
parser.add_argument('-mti', '--max-train-iters', default=np.inf, type=int,
					help='number of training iterations per epoch before cutting off (default: infinite)')

parser.add_argument('--wandb', action='store_true', help='use wandb logging')




#...............model args begin ..........................
parser.add_argument('-a', '--arch', metavar='ARCH', default='UNet_4down_aps',
					help='U-Net arch. Possible options: UNet_3down, UNet_4down, UNet_4down_InstanceNorm, UNet_4down_lpf, UNet_3down_lpf, UNet_4down_aps, UNet_3down_aps, auto_3down, auto_4down')

parser.add_argument('--in_channels', default=1, type = int,
					help='Input U-Net channels')

parser.add_argument('--out_channels', default=1, type = int,
					help='Output U-Net channels')

parser.add_argument('--inner_channels_list', default=[64, 128, 256, 512, 1024], type = int, nargs='+',
					help='Inner layer channels')


parser.add_argument('--bilinear', action='store_true',
					help='Bilinear mode in U-Net upsample flag.')

parser.add_argument('--padding_mode', default = 'circular', type = str,
					help='Padding mode: circular, zeros.')

parser.add_argument('--filter_size', default = 1, type = int,
					help='Filter size')

parser.add_argument('--sinc_mode', action='store_true',
					help='Turn on sinc filter for LPF. Currently not supported for APS. Avoid using for now.')


#...............dataset args..........................





lowest_loss = 1e6


def main():
	args = parser.parse_args()
	
	if args.visible_devices !='all':
		os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_devices
	
	if(not os.path.exists(args.out_dir)):
		os.mkdir(args.out_dir)

	if args.seed is not None:
		random.seed(args.seed)
		torch.manual_seed(args.seed)
#         cudnn.deterministic = True
		warnings.warn('You have chosen to seed training. '
					  'This will turn on the CUDNN deterministic setting, '
					  'which can slow down your training considerably! '
					  'You may see unexpected behavior when restarting '
					  'from checkpoints.')

	if args.gpu is not None:
		warnings.warn('You have chosen a specific GPU. This will completely '
					  'disable data parallelism.')

	if args.dist_url == "env://" and args.world_size == -1:
		args.world_size = int(os.environ["WORLD_SIZE"])

	args.distributed = args.world_size > 1 or args.multiprocessing_distributed

	ngpus_per_node = torch.cuda.device_count()
	if args.multiprocessing_distributed:
		# Since we have ngpus_per_node processes per node, the total world_size
		# needs to be adjusted accordingly
		args.world_size = ngpus_per_node * args.world_size
		# Use torch.multiprocessing.spawn to launch distributed processes: the
		# main_worker process function
		mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
	else:
		# Simply call main_worker function
		main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):

	global lowest_loss
	args.gpu = gpu

	if args.gpu is not None:
		print("Use GPU: {} for training".format(args.gpu))

	if args.distributed:
		if args.dist_url == "env://" and args.rank == -1:
			args.rank = int(os.environ["RANK"])
		if args.multiprocessing_distributed:
			# For multiprocessing distributed training, rank needs to be the
			# global rank among all the processes
			args.rank = args.rank * ngpus_per_node + gpu
		dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
								world_size=args.world_size, rank=args.rank)

	# create model
	print("=> creating model '{}'".format(args.arch))




	# fixed seed for model weight initialization
	np.random.seed(0)
	random.seed(0)
	torch.manual_seed(0)


	model_dict = {'in_channels': args.in_channels,
		'out_channels': args.out_channels,
		'inner_channels_list': args.inner_channels_list,
		'bilinear': args.bilinear,
		'padding_mode': args.padding_mode}

	if(args.arch.split('_')[-1]=='lpf'): # LPF model
		
		model_dict['sinc_mode'] = args.sinc_mode
		model_dict['filter_size'] = args.filter_size

		model = unet_lpf.__dict__[args.arch](**model_dict)
		print('Low pass filter size used: ', args.filter_size)
															  
	elif(args.arch.split('_')[-1]=='aps'): # APS models
		
		model_dict['filter_size'] = args.filter_size
		model = unet_aps.__dict__[args.arch](**model_dict)
		print('Low pass filter size used: ', args.filter_size)

	
	else: # baseline model
		model = baseline_unet.__dict__[args.arch](**model_dict)

	
	distributed_bool = (args.distributed ==True)

	if args.distributed:
		# For multiprocessing distributed, DistributedDataParallel constructor
		# should always set the single device scope, otherwise,
		# DistributedDataParallel will use all available devices.

		# distributed_bool = True


		if args.gpu is not None:
			torch.cuda.set_device(args.gpu)
			model.cuda(args.gpu)
			# When using a single GPU per process and per
			# DistributedDataParallel, we need to divide the batch size
			# ourselves based on the total number of GPUs we have
			args.batch_size = int(args.batch_size / ngpus_per_node)
			args.workers = int(args.workers / ngpus_per_node)
			model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
		else:
			model.cuda()
			# DistributedDataParallel will divide and allocate batch_size to all
			# available GPUs if device_ids are not set
			model = torch.nn.parallel.DistributedDataParallel(model)
	elif args.gpu is not None:
		# distributed_bool = False

		torch.cuda.set_device(args.gpu)
		model = model.cuda(args.gpu)
	else:
		# distributed_bool = False
		# DataParallel will divide and allocate batch_size to all available GPUs
		if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
			model.features = torch.nn.DataParallel(model.features)
			model.cuda()
		else:
			model = torch.nn.DataParallel(model).cuda()

	# define loss function (criterion) and optimizer

	criterion = eval(args.criterion)().cuda(args.gpu)



	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, 
		weight_decay=args.weight_decay)

	if args.evaluate or args.evaluate_save_unshifted or args.evaluate_random_shift or args.evaluate_equivariance_metrics or args.evaluate_psnr_diff_random_shifts or args.evaluate_vert_flip_diff_psnr_random_shifts:
		args.batch_size=1


	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			model.load_state_dict(checkpoint['state_dict'], strict=False)
			if('optimizer' in checkpoint.keys()): # if no optimizer, then only load weights
				args.start_epoch = checkpoint['epoch']
				lowest_loss = checkpoint['lowest_loss']
				# if args.gpu is not None:
					# best_acc1 may be from a checkpoint from a different GPU
					# lowest_loss = lowest_loss.to(args.gpu)
				optimizer.load_state_dict(checkpoint['optimizer'])
			else:
				print('  No optimizer saved')

			print("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	cudnn.benchmark = True

	# Data loading code

	dataset_setting_dict = {'batch_size': args.batch_size,
						'dataset_dir': args.data, 
						'num_workers': args.workers,
						'distributed_bool': distributed_bool,
						'train_sample_complexity_frac': args.train_sample_complexity_frac}


	
	if args.train_sample_complexity_frac<1:
		# fixing seed to ensure same train split while training every model
		np.random.seed(0)
		random.seed(0)

	dataloaders, train_sampler = get_dataloaders_fastmri(** dataset_setting_dict, include_test = False )

	if(args.val_debug): # debug mode - train on val set for faster epochs
		train_loader = val_loader


	if args.evaluate:

		args.batch_size = 1
		print('Batch size chosen to 1 for evaluation.')
		
		model.eval()
		
		print('==>Evaluating model for PSNR on unshifted images.')
		metrics = evaluate_psnr_streamlined(dataloaders, model, args)
		print(metrics)

		evaluation_type = 'unshifted_'+args.eval_phase+'_images'
		if args.evaluate_acquisition is not None:
			evaluation_type+= '_'+ args.evaluate_acquisition 

		save_metrics(metrics, args, evaluation_type = evaluation_type)

		return

	if args.evaluate_random_shift:

		args.batch_size = 1
		print('Batch size chosen to 1 for evaluation.')
		
		model.eval()
		#seed fixed so that same shifts are used during eval for each model.
		np.random.seed(0)
		random.seed(0)
		torch.manual_seed(0)
		
		print('==>Evaluating model for PSNR on random shifted images.')
		metrics = evaluate_psnr_streamlined_shift(dataloaders, model, args)
		print(metrics)

		evaluation_type = 'random_shifted_'+args.eval_phase+'_images'
		if args.evaluate_acquisition is not None:
			evaluation_type+= '_'+ args.evaluate_acquisition 

		save_metrics(metrics, args, evaluation_type = evaluation_type)

		return

	if args.evaluate_save_unshifted:

		print('Unshifted images validation begins.==>')
		outputs, observed_signals = evaluate_save_unshifted(dataloaders, model, args)
		
		if not os.path.isdir(os.path.join(args.out_dir, 'unshifted_reconstruction_files')):
			os.mkdir(os.path.join(args.out_dir, 'unshifted_reconstruction_files'))


		print('Saving unshifted reconstructions and inputs.==>')
		pickle.dump(outputs, open(os.path.join(args.out_dir, 'unshifted_reconstruction_files','outputs.p'), 'wb'))
		pickle.dump(observed_signals, open(os.path.join(args.out_dir, 'unshifted_reconstruction_files','observed_signals.p'), 'wb'))

		return 

	if args.evaluate_out_of_dist_imagenet_equivariance:

		# from utils.load_lodopab_ct import get_dataloaders_ct
		# ct_data_path = '/raid/datasets/ct_lodopab/'
		# ct_dataset_image_crop = 352

		imagenet_val_path = os.path.join('/raid/datasets/imagenet/val')

		mean=[0.485, 0.456, 0.406]
		std=[0.229, 0.224, 0.225]
		normalize = torchvision.transforms.Normalize(mean=mean, std=std)

		val_dataset = torchvision.datasets.ImageFolder(imagenet_val_path, torchvision.transforms.Compose([
			torchvision.transforms.Resize(256),
			torchvision.transforms.CenterCrop(224),
			torchvision.transforms.ToTensor(),
			normalize,
			torchvision.transforms.Grayscale(),
		]))

		val_dataset = torch.utils.data.Subset(val_dataset, np.arange(args.imagenet_out_of_dist_num))

		imagenet_val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

	   
		model.eval()
		np.random.seed(0)
		random.seed(0)
		torch.manual_seed(0)

		metrics = evaluate_imagenet_out_of_dist_equivariance_metrics(imagenet_val_loader, model, args)
		print(metrics)

		evaluation_type = 'imagenet_out_of_dist_equivariance_'+str(args.imagenet_out_of_dist_num)+'_images'
		# if args.evaluate_acquisition is not None:
		#     evaluation_type+= '_'+args.evaluate_acquisition 

		save_metrics(metrics, args, evaluation_type = evaluation_type)

		return 


	if args.evaluate_out_of_dist_ct_equivariance:

		from utils.load_lodopab_ct import get_dataloaders_ct
		ct_data_path = '/raid/datasets/ct_lodopab/'
		ct_dataset_image_crop = 352

		ct_dataset_dict = {'batch_size': 1, 'num_workers': args.workers, 'include_validation': True, 
		'cache_dir': os.path.join(ct_data_path, 'cache_fbp_lodopab'), 'dataset_image_center_crop': ct_dataset_image_crop}

		ct_dataloaders = get_dataloaders_ct(**ct_dataset_dict)

		model.eval()
		np.random.seed(0)
		random.seed(0)
		torch.manual_seed(0)

		metrics = evaluate_ct_out_of_dist_equivariance_metrics(ct_dataloaders['test'], model, args)
		print(metrics)

		evaluation_type = 'ct_out_of_dist_equivariance_test_images'
		# if args.evaluate_acquisition is not None:
		#     evaluation_type+= '_'+args.evaluate_acquisition 

		save_metrics(metrics, args, evaluation_type = evaluation_type)

		return 


	if args.evaluate_out_of_dist_wgn_equivariance:

		model.eval()
		np.random.seed(0)
		random.seed(0)
		torch.manual_seed(0)

		metrics = evaluate_wgn_equivariance_metrics(model, args)
		print(metrics)



		evaluation_type = 'pure_wgn_out_of_dist_equivariance_'+str(args.wgn_out_of_dist_num)+'_images'
		# if args.evaluate_acquisition is not None:
		#     evaluation_type+= '_'+args.evaluate_acquisition 

		save_metrics(metrics, args, evaluation_type = evaluation_type)

		return 






	if args.evaluate_equivariance_metrics:

		args.batch_size = 1
		print('Batch size chosen to 1 for evaluation.')

		model.eval()
		np.random.seed(0)
		random.seed(0)
		torch.manual_seed(0)

		print('==>Evaluating metrics between shift(f(x)) and f(shift(x)), where f is the U-Net.')
		metrics = evaluate_equivariance_metrics(dataloaders, model, args)
		print(metrics)

		evaluation_type = 'equivariance_'+args.eval_phase+'_images'
		if args.evaluate_acquisition is not None:
			evaluation_type+= '_'+args.evaluate_acquisition 

		save_metrics(metrics, args, evaluation_type = evaluation_type)

		return 


	if args.evaluate_psnr_diff_random_shifts:

		args.batch_size = 1
		print('Batch size chosen to 1 for evaluation.')

		model.eval()
		np.random.seed(0)
		random.seed(0)
		torch.manual_seed(0)

		print('==> Evaluating absolute difference between PSNR obtained with image and its random shift.')

		psnr_unshifted_list = []
		psnr_shifted_list = []

		for i in range(args.psnr_diff_eval_num_shifts):
			print('\nShift pass: ', i)
			psnr_unshifted_list_curr, psnr_shifted_list_curr = evaluate_psnr_diff_random_shifts(dataloaders, model, args)

			psnr_unshifted_list+=psnr_unshifted_list_curr
			psnr_shifted_list+=psnr_shifted_list_curr


		mean_abs_diff = np.mean(np.abs(np.array(psnr_unshifted_list) - np.array(psnr_shifted_list)))
		max_abs_diff = np.max(np.abs(np.array(psnr_unshifted_list) - np.array(psnr_shifted_list)))

		eval_result_string = 'Mean PSNR on unshifted images: '+str(np.mean(psnr_unshifted_list))+'\n'
		eval_result_string+= 'Mean PSNR on randomly shifted images: '+str(np.mean(psnr_shifted_list))+'\n'
		eval_result_string+= 'Mean of abs difference between psnr_unshift and psnr_shift: '+str(mean_abs_diff)+ '\n'
		eval_result_string+= 'Max of abs difference between psnr_unshift and psnr_shift: '+str(max_abs_diff) + '\n'
		eval_result_string+='Number of passes through the dataset for evaluation: '+str(args.psnr_diff_eval_num_shifts)+'\n\n'

		eval_result = {'psnr_unshifted_list': psnr_unshifted_list, 'psnr_shifted_list':psnr_shifted_list, 'psnr_diff_eval_num_shifts': args.psnr_diff_eval_num_shifts}

		evaluation_type = 'diff_psnr_shift_'+args.eval_phase+'_images'+'_shift_passes_'+str(args.psnr_diff_eval_num_shifts)
		if args.evaluate_acquisition is not None:
			evaluation_type+= '_'+args.evaluate_acquisition 

		print(eval_result_string)
		pickle_dump_and_write(object_to_write = eval_result_string, object_to_pickle_save = eval_result, path = args.out_dir, file_name = 'evaluate_'+evaluation_type)

		return 


	if args.evaluate_vert_flip_diff_psnr_random_shifts:

		args.batch_size = 1
		print('Batch size chosen to 1 for evaluation.')

		model.eval()
		np.random.seed(0)
		random.seed(0)
		torch.manual_seed(0)

		print('==> Evaluating absolute difference between PSNR obtained with vertically flipped images and its random shift.')

		psnr_unshifted_list = []
		psnr_shifted_list = []

		for i in range(args.psnr_diff_eval_num_shifts):
			print('\nShift pass: ', i)
			psnr_unshifted_list_curr, psnr_shifted_list_curr = evaluate_psnr_diff_random_shifts(dataloaders, model, args)

			psnr_unshifted_list+=psnr_unshifted_list_curr
			psnr_shifted_list+=psnr_shifted_list_curr


		mean_abs_diff = np.mean(np.abs(np.array(psnr_unshifted_list) - np.array(psnr_shifted_list)))
		max_abs_diff = np.max(np.abs(np.array(psnr_unshifted_list) - np.array(psnr_shifted_list)))

		eval_result_string = 'Mean PSNR on unshifted images: '+str(np.mean(psnr_unshifted_list))+'\n'
		eval_result_string+= 'Mean PSNR on randomly shifted images: '+str(np.mean(psnr_shifted_list))+'\n'
		eval_result_string+= 'Mean of abs difference between psnr_unshift and psnr_shift: '+str(mean_abs_diff)+ '\n'
		eval_result_string+= 'Max of abs difference between psnr_unshift and psnr_shift: '+str(max_abs_diff) + '\n'
		eval_result_string+='Number of passes through the dataset for evaluation: '+str(args.psnr_diff_eval_num_shifts)+'\n\n'

		eval_result = {'psnr_unshifted_list': psnr_unshifted_list, 'psnr_shifted_list':psnr_shifted_list, 'psnr_diff_eval_num_shifts': args.psnr_diff_eval_num_shifts}

		evaluation_type = 'diff_psnr_shift_vertically_flipped_'+args.eval_phase+'_images'+'_shift_passes_'+str(args.psnr_diff_eval_num_shifts)
		if args.evaluate_acquisition is not None:
			evaluation_type+= '_'+args.evaluate_acquisition 

		print(eval_result_string)
		pickle_dump_and_write(object_to_write = eval_result_string, object_to_pickle_save = eval_result, path = args.out_dir, file_name = 'evaluate_'+evaluation_type)

		return 


	if args.evaluate_out_of_dist_noise_equivariance:

		args.batch_size = 1
		print('Batch size chosen to 1 for evaluation.')

		model.eval()
		np.random.seed(0)
		random.seed(0)
		torch.manual_seed(0)
		#seed used to get the same noise and shift for each model 


		print('\nEvaluating equivariance metrics on out of distribution images. ==>')
		print('Eval noise level: ', args.eval_noise_level, '\n')

		metrics = evaluate_out_of_dist_equivariance_metrics(dataloaders, model, args)

		print(metrics)

		evaluation_type = 'out_of_dist_equivariance_'+args.eval_phase+'_images_noise_'+str(args.eval_noise_level)
		if args.evaluate_acquisition is not None:
			evaluation_type+= '_'+args.evaluate_acquisition 

		save_metrics(metrics, args, evaluation_type = evaluation_type)

		return


	if args.evaluate_out_of_dist_noise_unshifted_metrics:

		args.batch_size = 1
		print('Batch size chosen to 1 for evaluation.')

		model.eval()
		np.random.seed(0)
		random.seed(0)
		torch.manual_seed(0)
		#seed used to get the same noise for each model

		print('\nEvaluating performance metrics on unshifted images with added noise. ==>')
		print('Eval noise level: ', args.eval_noise_level, '\n')

		metrics = evaluate_out_of_dist_unshifted_metrics(dataloaders, model, args)
		print(metrics)
		evaluation_type = 'out_of_dist_unshifted_'+args.eval_phase+'_images_noise_'+str(args.eval_noise_level)
		if args.evaluate_acquisition is not None:
			evaluation_type+= '_'+args.evaluate_acquisition

		save_metrics(metrics, args, evaluation_type = evaluation_type) 


		return 


	if args.evaluate_out_of_dist_psnr_diff_with_shift:

		args.batch_size = 1
		print('Batch size chosen to 1 for evaluation.')

		model.eval()
		np.random.seed(0)
		random.seed(0)
		torch.manual_seed(0)

		print('==> Evaluating absolute difference between PSNR obtained with noisy image and its random shift.')

		psnr_unshifted_list, psnr_shifted_list = evaluate_out_of_dist_psnr_diff_with_shift(dataloaders, model, args)

		mean_abs_diff = np.mean(np.abs(np.array(psnr_unshifted_list) - np.array(psnr_shifted_list)))
		median_abs_diff = np.mean(np.abs(np.array(psnr_unshifted_list) - np.array(psnr_shifted_list)))
		max_abs_diff = np.max(np.abs(np.array(psnr_unshifted_list) - np.array(psnr_shifted_list)))

		eval_result_string = 'Mean PSNR on unshifted images: '+str(np.mean(psnr_unshifted_list))+'\n'
		eval_result_string+= 'Mean PSNR on randomly shifted images: '+str(np.mean(psnr_shifted_list))+'\n'
		eval_result_string+= 'Mean of abs difference between psnr_unshift and psnr_shift: '+str(mean_abs_diff)+ '\n'
		eval_result_string+= 'Median of abs difference between psnr_unshift and psnr_shift: '+str(median_abs_diff)+ '\n'
		eval_result_string+= 'Max of abs difference between psnr_unshift and psnr_shift: '+str(max_abs_diff) + '\n\n'

		eval_result = {'psnr_unshifted_list': psnr_unshifted_list, 'psnr_shifted_list':psnr_shifted_list}

		evaluation_type = 'out_of_dist_diff_psnr_shift_'+args.eval_phase+'_images_'+str(args.eval_noise_level)
		if args.evaluate_acquisition is not None:
			evaluation_type+= '_'+args.evaluate_acquisition 

		path = os.path.join(args.out_dir, 'out_of_dist_exps')
		
		if not os.path.isdir(path):
			os.mkdir(path)


		print(eval_result_string)
		pickle_dump_and_write(object_to_write = eval_result_string, object_to_pickle_save = eval_result, path = path, file_name = 'evaluate_'+evaluation_type)

		return 



   
	if(args.cos_lr):

		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min= 5e-6)
		for epoch in range(args.start_epoch):
			scheduler.step()

	else:
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size =args.lr_step , gamma=args.lr_decay_factor)
		for epoch in range(args.start_epoch):
			scheduler.step()

	pickle_dump_and_write(args, args, path = args.out_dir, file_name = 'config_args')
	
	
	np.random.seed(0)
	random.seed(0)
	torch.manual_seed(0)

	print('Training begins==>')
	for epoch in range(args.start_epoch, args.epochs):
		if args.distributed:
			train_sampler.set_epoch(epoch)

		

		if(args.wandb):
			wandb.log({'learning_rate': optimizer.param_groups[0]['lr']},
					  commit=False)

		# train for one epoch
		train(dataloaders['train'], model, criterion, optimizer, epoch, args)

		scheduler.step()
		# print('[%03d] %.5f'%(epoch, scheduler.get_last_lr()[0]))

		# evaluate on validation set
		loss = validate_loss(dataloaders['validation'], model, criterion, args)

		# remember best scenario (lowest loss) and save checkpoint
		is_best = loss < lowest_loss
		lowest_loss = min(lowest_loss, loss)

		if not args.multiprocessing_distributed or (args.multiprocessing_distributed
				and args.rank % ngpus_per_node == 0):
			save_checkpoint({
				'epoch': epoch + 1,
				'arch': args.arch,
				'state_dict': model.state_dict(),
				'lowest_loss': lowest_loss,
				'optimizer' : optimizer.state_dict(),
			}, is_best, epoch, out_dir=args.out_dir)


def train(train_loader, model, criterion, optimizer, epoch, args):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	# psnr1 = AverageMeter()

	# switch to train mode
	model.train()

	loss = 0

	end = time.time()
	accum_track = 0
	optimizer.zero_grad()
	for i, (x, d, _, _, _, _, _ ) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)
		x = x.unsqueeze(1)
		d = d.unsqueeze(1)
		
		# if args.gpu is not None:
		x = x.cuda(args.gpu, non_blocking=True)
		d = d.cuda(args.gpu, non_blocking=True)
		
		if args.circular_data_aug:
			shift = np.random.randint(-args.max_shift, args.max_shift,size=2)
			x = torch.roll(x, shifts = (shift[0], shift[1]) , dims = (2, 3) )
			d = torch.roll(d, shifts = (shift[0], shift[1]) , dims = (2, 3) )

		# if args.random_crop_data_aug:
		#     shift = np.random.randint(-args.max_shift, args.max_shift,size=2)
		#     x = torch.nn.functional.pad(x, pad = (args.max_shift, args.max_shift, args.max_shift, args.max_shift))
		#     d = torch.nn.functional.pad(d, pad = (args.max_shift, args.max_shift, args.max_shift, args.max_shift))



		# compute output
		output = model(x)
		output = output.clamp(-6, 6)
		loss = criterion(output, d)

		losses.update(loss.item(), x.size(0))

		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)            

		accum_track+=1
		if(accum_track==args.batch_accum):
			optimizer.step()
			accum_track = 0
			optimizer.zero_grad()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
				   epoch, i, len(train_loader), batch_time=batch_time,
				   data_time=data_time, loss=losses))

			string = '\n\nEpoch: [{0}][{1}/{2}]\t''Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t''Data {data_time.val:.3f} ({data_time.avg:.3f})\t''Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses)
				  

			f = open(args.out_dir+'/train_log.txt', 'a')
			f.write( string )
			f.close()

			if(args.wandb):
				import wandb
				global_step = i + (epoch * len(train_loader))
				wandb.log(
					{
						'train_loss': losses.val,
						'train_avg_loss': losses.avg,
						'epoch': 1.*global_step/len(train_loader), 
					},
					step=global_step)

		if(i > args.max_train_iters):
			break

def validate_loss(val_loader, model, criterion, args):
	batch_time = AverageMeter()
	losses = AverageMeter()
	
	covariance_losses = AverageMeter()

	# switch to evaluate mode
	model.eval()

	with torch.no_grad():
		end = time.time()

		for i, (x, d, mean, std, _, _, _ ) in enumerate(val_loader):
			x = x.unsqueeze(1)
			d = d.unsqueeze(1)

			# if args.gpu is not None:
			x = x.cuda(args.gpu, non_blocking=True)
			d = d.cuda(args.gpu, non_blocking=True)
				

			# compute output
			output = model(x)
			output = output.clamp(-6, 6)
			loss = criterion(output, d)

			#compute equivariance mse
			shift = np.random.randint(-args.max_shift, args.max_shift, 2)

			shifted_x = torch.roll(x, shifts = (shift[0], shift[1]), dims = (2,3))
			output_shifted_x = model(shifted_x)
			output_shifted_x = output_shifted_x.clamp(-6, 6)

			shifted_output = torch.roll(output, shifts = (shift[0], shift[1]), dims = (2,3))

			covariance_loss = criterion(output_shifted_x, shifted_output)

			# update losses
			losses.update(loss.item(), x.size(0))
			covariance_losses.update(covariance_loss.item(), x.size(0))    

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.print_freq == 0:
				print('Test: [{0}/{1}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Covariance loss {cov_loss.val:.4f} ({cov_loss.avg:.4f})'.format(
					   i, len(val_loader), batch_time=batch_time, loss=losses, cov_loss = covariance_losses))

					  
				string = '\n\nTest: [{0}/{1}]\t''Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t''Loss {loss.val:.4f} ({loss.avg:.4f})\t''Covariance loss {cov_loss.val:.4f} ({cov_loss.avg:.4f})'.format(i, len(val_loader), batch_time=batch_time, loss=losses, cov_loss = covariance_losses)

				f = open(args.out_dir+'/val_log.txt', 'a')
				f.write( string )
				f.close()
				

		if args.wandb:
			import wandb
			wandb.log(
				{
					'val_avg_loss': losses.avg,
					# 'val_avg_psnr1': psnr1.avg,
				},
				commit=False)

	return losses.avg

def evaluate_psnr_streamlined(dataloaders, model, args):
	
	evaluate_acquisition_names = {'PD': 'CORPD_FBK', 'PDFS': 'CORPDFS_FBK'}

	if args.eval_phase == 'validation':
		target_path = Path(os.path.join(args.data, 'singlecoil_val'))
	else:
		raise Exception('Other evaluation phases like train and test currently not supported.')
	

	outputs = defaultdict(list)
	with torch.no_grad():
		with tqdm(dataloaders[args.eval_phase]) as pbar:
			for obs, gt,  mean, std, fname, slice_num, max_value in pbar:
				fname = fname[0]

				if len(obs.shape)==3:
					obs = obs.unsqueeze(1)
					gt = gt.unsqueeze(1)

				mean = mean.unsqueeze(1).unsqueeze(2)
				std = std.unsqueeze(1).unsqueeze(2)

				reco = model(obs.cuda()).cpu().clamp(-6, 6)
					
				if reco.max() > 6:
					print(reco.max())

				# undo the instance-normalized the output and target
				trans_reco = (reco * std + mean).detach().numpy().squeeze()

				# collect slices into the volume it belongs to
				outputs[fname].append((slice_num.numpy(), trans_reco ))
				
	# numpy stack output images for individual fnames
	for fname in outputs.keys():
		outputs[fname] = np.stack([out for _, out in sorted(outputs[fname])])
		
	#evaluate PSNR with original targets loaded from file
	
	METRIC_FUNCS = dict(
	NMSE=nmse,
	PSNR=psnr, 
	SSIM=ssim)

	metrics = Metrics(METRIC_FUNCS)
	print('Images reconstructed and gathered. Metrics (NMSE, PSNR, SSIM) calculation begins==>')
	for tgt_file in tqdm(target_path.iterdir()):
		
		fname = tgt_file.name
		target = h5py.File(tgt_file, 'r')
		recons = outputs[fname]
		
		if args.evaluate_acquisition and evaluate_acquisition_names[args.evaluate_acquisition] != target.attrs['acquisition']:
			continue

		target = target['reconstruction_esc'][()]
		target = fastmri_transforms.center_crop(target, (recons.shape[-1], recons.shape[-1]))
		metrics.push(target, recons)
		
	
	return metrics


def evaluate_save_unshifted(dataloaders, model, args):

	# evaluate_acquisition_names = {'PD': 'CORPD_FBK', 'PDFS': 'CORPDFS_FBK'}
	#save all reconstructions irrespective of PD/PDFS

	if args.batch_size!=1:
		raise Exception('Batch size not 1 for evaluation')

	if args.eval_phase == 'validation':
		target_path = Path(os.path.join(args.data, 'singlecoil_val'))
	else:
		raise Exception('Other evaluation phases like train and test currently not supported.')
	

	outputs = defaultdict(list)
	observed_signals = defaultdict(list)
	with torch.no_grad():
		with tqdm(dataloaders[args.eval_phase]) as pbar:
			for obs, gt,  mean, std, fname, slice_num, max_value in pbar:
				fname = fname[0]

				if len(obs.shape)==3:
					obs = obs.unsqueeze(1)
					gt = gt.unsqueeze(1)

				mean = mean.unsqueeze(1).unsqueeze(2)
				std = std.unsqueeze(1).unsqueeze(2)

				reco = model(obs.cuda()).cpu().clamp(-6, 6)
					
				if reco.max() > 6:
					print(reco.max())

				# undo the instance-normalized the output and observed
				trans_reco = (reco * std + mean).detach().numpy().squeeze()
				trans_obs = (obs * std + mean).detach().numpy().squeeze()

				# collect slices into the volume it belongs to
				outputs[fname].append((slice_num.numpy(), trans_reco ))
				observed_signals[fname].append((slice_num.numpy(), trans_obs ))
				
	# numpy stack output images for individual fnames
	for fname in outputs.keys():
		outputs[fname] = np.stack([out for _, out in sorted(outputs[fname])])
		observed_signals[fname] = np.stack([obs for _, obs in sorted(observed_signals[fname])])
		
	#evaluate PSNR with original targets loaded from file

	return outputs, observed_signals
	
	
	# metrics = Metrics(METRIC_FUNCS)
	# print('Images reconstructed and gathered. Metrics (NMSE, PSNR, SSIM) calculation begins==>')
	# for tgt_file in tqdm(target_path.iterdir()):
		
	#     fname = tgt_file.name
	#     target = h5py.File(tgt_file, 'r')
	#     recons = outputs[fname]
		
	#     if args.evaluate_acquisition and evaluate_acquisition_names[args.evaluate_acquisition] != target.attrs['acquisition']:
	#         continue

	#     target = target['reconstruction_esc'][()]
	#     target = fastmri_transforms.center_crop(target, (recons.shape[-1], recons.shape[-1]))
	#     metrics.push(target, recons)
		
	
	return metrics





def evaluate_psnr_streamlined_shift(dataloaders, model, args):
	
	evaluate_acquisition_names = {'PD': 'CORPD_FBK', 'PDFS': 'CORPDFS_FBK'}

	if args.eval_phase == 'validation':
		target_path = Path(os.path.join(args.data, 'singlecoil_val'))
	else:
		raise Exception('Other evaluation phases like train and test currently not supported.')
	

	outputs = defaultdict(list)
	shifts = defaultdict(list)
	with torch.no_grad():
		with tqdm(dataloaders[args.eval_phase]) as pbar:
			for obs, gt,  mean, std, fname, slice_num, max_value in pbar:
				fname = fname[0]
				if len(obs.shape)==3:
					obs = obs.unsqueeze(1)
					gt = gt.unsqueeze(1)
	
				mean = mean.unsqueeze(1).unsqueeze(2)
				std = std.unsqueeze(1).unsqueeze(2)
				
				#construct shift for fname
				if fname in shifts.keys():
					shift = shifts[fname]
				else:
					shift = list(np.random.randint(low = -args.max_shift, high=args.max_shift, size=2))
					shifts[fname] = shift
				
				reco = model(torch.roll(obs.cuda(), shifts = shift, dims = (2,3))).cpu().clamp(-6, 6)
					
				# undo the instance-normalized the output and target
				translated_reco = (reco * std + mean).detach().numpy().squeeze()

				# collect slices into the volume it belongs to
				outputs[fname].append((slice_num.numpy(), translated_reco ))
				
	# numpy stack output images for individual fnames
	for fname in outputs.keys():
		outputs[fname] = np.stack([out for _, out in sorted(outputs[fname])])
		
	#evaluate PSNR with original targets loaded from file
	
	METRIC_FUNCS = dict(
	NMSE=nmse,
	PSNR=psnr, 
	SSIM=ssim)

	metrics = Metrics(METRIC_FUNCS)
	print('Images reconstructed and gathered. Metrics (NMSE, PSNR, SSIM) calculation begins==>')
	for tgt_file in tqdm(target_path.iterdir()):
		
		fname = tgt_file.name
		target = h5py.File(tgt_file, 'r')
		recons = outputs[fname]
		
		if args.evaluate_acquisition and evaluate_acquisition_names[args.evaluate_acquisition] != target.attrs['acquisition']:
			continue

		target = target['reconstruction_esc'][()]
		target = fastmri_transforms.center_crop(target, (recons.shape[-1], recons.shape[-1]))
		target_shift = np.roll(target, shift = shifts[fname], axis = (1, 2))
		
		metrics.push(target_shift, recons)
		
	
	return metrics



def evaluate_psnr_diff_random_shifts(dataloaders, model, args):
	
	evaluate_acquisition_names = {'PD': 'CORPD_FBK', 'PDFS': 'CORPDFS_FBK'}

	if args.eval_phase == 'validation':
		target_path = Path(os.path.join(args.data, 'singlecoil_val'))
	else:
		raise Exception('Other evaluation phases like train and test currently not supported.')
	

	outputs = defaultdict(list)
	outputs_shifted_inputs = defaultdict(list)

	shifts = defaultdict(list)
	with torch.no_grad():
		with tqdm(dataloaders[args.eval_phase]) as pbar:
			for obs, gt,  mean, std, fname, slice_num, max_value in pbar:
				fname = fname[0]
				if len(obs.shape)==3:
					obs = obs.unsqueeze(1)
					gt = gt.unsqueeze(1)
	
				mean = mean.unsqueeze(1).unsqueeze(2)
				std = std.unsqueeze(1).unsqueeze(2)

				if args.evaluate_vert_flip_diff_psnr_random_shifts:
					obs = torch.flip(obs, dims = [2])
					gt = torch.flip(gt, dims = [2])
				
				#construct shift for fname
				if fname in shifts.keys():
					shift = shifts[fname]
				else:
					shift = list(np.random.randint(low = -args.max_shift, high=args.max_shift, size=2))
					shifts[fname] = shift
				
				reco = model( obs.cuda()).cpu().clamp(-6, 6)
				reco_shifted_inp = model(torch.roll(obs.cuda(), shifts = shift, dims = (2,3))).cpu().clamp(-6, 6)
					
				# undo the instance-normalized the output and target
				translated_reco = (reco * std + mean).detach().numpy().squeeze()
				translated_reco_shifted_inp = (reco_shifted_inp * std + mean).detach().numpy().squeeze()

				# collect slices into the volume it belongs to
				outputs[fname].append((slice_num.numpy(), translated_reco ))
				outputs_shifted_inputs[fname].append((slice_num.numpy(), translated_reco_shifted_inp ))

				
	# numpy stack output images for individual fnames
	for fname in outputs.keys():
		outputs[fname] = np.stack([out for _, out in sorted(outputs[fname])])
		outputs_shifted_inputs[fname] = np.stack([out for _, out in sorted(outputs_shifted_inputs[fname])])

	#evaluate PSNR with original targets loaded from file

	psnr_unshifted_list = []
	psnr_shifted_list = []


	# metrics = Metrics(METRIC_FUNCS)
	print('Images reconstructed and gathered. PSNR calculation for unshifted and randomly shifted images begins==>')
	for tgt_file in tqdm(target_path.iterdir()):
		
		fname = tgt_file.name
		target = h5py.File(tgt_file, 'r')
		out = outputs[fname]
		out_shifted_inp = outputs_shifted_inputs[fname]
		
		if args.evaluate_acquisition and evaluate_acquisition_names[args.evaluate_acquisition] != target.attrs['acquisition']:
			continue

		target = target['reconstruction_esc'][()]
		target = fastmri_transforms.center_crop(target, (out.shape[-2], out.shape[-1]))

		if args.evaluate_vert_flip_diff_psnr_random_shifts:
			target = np.flip(target, axis = 1)

		target_shift = np.roll(target, shift = shifts[fname], axis = (1, 2))

		psnr_unshifted = psnr(target, out)
		psnr_shifted = psnr(target_shift, out_shifted_inp)

		psnr_unshifted_list.append(psnr_unshifted)
		psnr_shifted_list.append(psnr_shifted)
		
	
	return psnr_unshifted_list, psnr_shifted_list

def evaluate_wgn_equivariance_metrics(model, args):

	if args.batch_size!=1:
		raise Exception('Batch size not 1 for PSNR and SSIM evaluation.')

	METRIC_FUNCS = dict(
	NMSE=nmse,
	PSNR=psnr, 
	SSIM=ssim)

	metrics = Metrics(METRIC_FUNCS)

	model.eval()
	with torch.no_grad():

		for i in tqdm(range(args.wgn_out_of_dist_num)):
			
			shift = list(np.random.randint(low = -args.max_shift, high=args.max_shift, size=2))

			obs = 1000*torch.randn(1, 1, 320, 320).cuda()
			shifted_obs = torch.roll(obs, shifts = shift, dims = (2,3))

			out = model(obs)
			out_shifted_inp = model(shifted_obs)

			shifted_out = torch.roll(out, shifts = shift, dims = (2,3))

			metrics.push(shifted_out[:, 0, :, :].detach().cpu().numpy(), out_shifted_inp[:, 0, :, :].detach().cpu().numpy())

	return metrics




def evaluate_ct_out_of_dist_equivariance_metrics(dataloader, model, args):

	if args.batch_size!=1:
		raise Exception('Batch size not 1 for PSNR and SSIM evaluation.')

	METRIC_FUNCS = dict(
	NMSE=nmse,
	PSNR=psnr, 
	SSIM=ssim)

	metrics = Metrics(METRIC_FUNCS)

	model.eval()
	with torch.no_grad():
		with tqdm(dataloader) as pbar:
			for obs, gt in pbar:

				# print(obs.shape)
				# print(gt.shape)

				shift = list(np.random.randint(low = -args.max_shift, high=args.max_shift, size=2))

				obs = obs.cuda()
				shifted_obs = torch.roll(obs, shifts = shift, dims = (2,3))

				out = model(obs)
				out_shifted_inp = model(shifted_obs)

				shifted_out = torch.roll(out, shifts = shift, dims = (2,3))

				# print(shifted_out.shape)
				# print(out_shifted_inp.shape)


				metrics.push(shifted_out[:, 0, :, :].detach().cpu().numpy(), out_shifted_inp[:, 0, :, :].detach().cpu().numpy())

	return metrics


def evaluate_imagenet_out_of_dist_equivariance_metrics(dataloader, model, args):

	if args.batch_size!=1:
		raise Exception('Batch size not 1 for PSNR and SSIM evaluation.')

	METRIC_FUNCS = dict(
	NMSE=nmse,
	PSNR=psnr, 
	SSIM=ssim)

	metrics = Metrics(METRIC_FUNCS)

	model.eval()
	with torch.no_grad():
		with tqdm(dataloader) as pbar:
			for obs, target in pbar:

				# print(obs.shape)
				# print(gt.shape)

				shift = list(np.random.randint(low = -args.max_shift, high=args.max_shift, size=2))

				obs = obs.cuda()
				shifted_obs = torch.roll(obs, shifts = shift, dims = (2,3))

				out = model(obs)
				out_shifted_inp = model(shifted_obs)

				shifted_out = torch.roll(out, shifts = shift, dims = (2,3))

				# print(shifted_out.shape)
				# print(out_shifted_inp.shape)


				metrics.push(shifted_out[:, 0, :, :].detach().cpu().numpy(), out_shifted_inp[:, 0, :, :].detach().cpu().numpy())

	return metrics



def evaluate_out_of_dist_psnr_diff_with_shift(dataloaders, model, args):
	
	evaluate_acquisition_names = {'PD': 'CORPD_FBK', 'PDFS': 'CORPDFS_FBK'}

	if args.eval_phase == 'validation':
		target_path = Path(os.path.join(args.data, 'singlecoil_val'))
	else:
		raise Exception('Other evaluation phases like train and test currently not supported.')
	

	outputs = defaultdict(list)
	outputs_shifted_inputs = defaultdict(list)

	shifts = defaultdict(list)
	with torch.no_grad():
		with tqdm(dataloaders[args.eval_phase]) as pbar:
			for obs, gt,  mean, std, fname, slice_num, max_value in pbar:
				fname = fname[0]
				if len(obs.shape)==3:
					obs = obs.unsqueeze(1)
					gt = gt.unsqueeze(1)
	
				mean = mean.unsqueeze(1).unsqueeze(2)
				std = std.unsqueeze(1).unsqueeze(2)
				
				#construct shift for fname
				if fname in shifts.keys():
					shift = shifts[fname]
				else:
					shift = list(np.random.randint(low = -args.max_shift, high=args.max_shift, size=2))
					shifts[fname] = shift
				
				obs = obs.cuda()
				obs += args.eval_noise_level*torch.randn(obs.shape).cuda()
				reco = model( obs).cpu().clamp(-6, 6)
				reco_shifted_inp = model(torch.roll(obs, shifts = shift, dims = (2,3))).cpu().clamp(-6, 6)
				obs = obs.cpu()
				
				# undo the instance-normalized the output and target
				translated_reco = (reco * std + mean).detach().numpy().squeeze()
				translated_reco_shifted_inp = (reco_shifted_inp * std + mean).detach().numpy().squeeze()

				# collect slices into the volume it belongs to
				outputs[fname].append((slice_num.numpy(), translated_reco ))
				outputs_shifted_inputs[fname].append((slice_num.numpy(), translated_reco_shifted_inp ))

				
	# numpy stack output images for individual fnames
	for fname in outputs.keys():
		outputs[fname] = np.stack([out for _, out in sorted(outputs[fname])])
		outputs_shifted_inputs[fname] = np.stack([out for _, out in sorted(outputs_shifted_inputs[fname])])

	#evaluate PSNR with original targets loaded from file

	psnr_unshifted_list = []
	psnr_shifted_list = []


	# metrics = Metrics(METRIC_FUNCS)
	print('Images reconstructed and gathered. PSNR calculation for unshifted and randomly shifted images begins==>')
	for tgt_file in tqdm(target_path.iterdir()):
		
		fname = tgt_file.name
		target = h5py.File(tgt_file, 'r')
		out = outputs[fname]
		out_shifted_inp = outputs_shifted_inputs[fname]
		
		if args.evaluate_acquisition and evaluate_acquisition_names[args.evaluate_acquisition] != target.attrs['acquisition']:
			continue

		target = target['reconstruction_esc'][()]
		target = fastmri_transforms.center_crop(target, (out.shape[-2], out.shape[-1]))
		target_shift = np.roll(target, shift = shifts[fname], axis = (1, 2))

		psnr_unshifted = psnr(target, out)
		psnr_shifted = psnr(target_shift, out_shifted_inp)

		psnr_unshifted_list.append(psnr_unshifted)
		psnr_shifted_list.append(psnr_shifted)
		
	
	return psnr_unshifted_list, psnr_shifted_list


def evaluate_equivariance_metrics(dataloaders, model, args):
	
	evaluate_acquisition_names = {'PD': 'CORPD_FBK', 'PDFS': 'CORPDFS_FBK'}

	if args.eval_phase == 'validation':
		target_path = Path(os.path.join(args.data, 'singlecoil_val'))
	else:
		raise Exception('Other evaluation phases like train and test currently not supported.')
	

	outputs = defaultdict(list)
	outputs_shifted_inp = defaultdict(list)

	shifts = defaultdict(list)

	with torch.no_grad():
		with tqdm(dataloaders[args.eval_phase]) as pbar:
			for obs, gt,  mean, std, fname, slice_num, max_value in pbar:
				fname = fname[0]
				if len(obs.shape)==3:
					obs = obs.unsqueeze(1)
					gt = gt.unsqueeze(1)
	
				mean = mean.unsqueeze(1).unsqueeze(2)
				std = std.unsqueeze(1).unsqueeze(2)
				
				#construct shift for fname
				if fname in shifts.keys():
					shift = shifts[fname]
				else:
					shift = list(np.random.randint(low = -args.max_shift, high=args.max_shift, size=2))
					shifts[fname] = shift
				
				reco = model(obs.cuda()).cpu().clamp(-6, 6)
				reco_shift = model(torch.roll(obs.cuda(), shifts = shift, dims = (2,3))).cpu().clamp(-6, 6)
					
				# undo the instance-normalized the output and target
				translated_reco = (reco * std + mean).detach().numpy().squeeze()
				translated_reco_shift = (reco_shift * std + mean).detach().numpy().squeeze()

				# collect slices into the volume it belongs to
				outputs[fname].append((slice_num.numpy(), translated_reco ))
				outputs_shifted_inp[fname].append((slice_num.numpy(), translated_reco_shift ))
				
	# numpy stack output images for individual fnames
	for fname in outputs.keys():
		outputs[fname] = np.stack([out for _, out in sorted(outputs[fname])])
		outputs_shifted_inp[fname] = np.stack([out_shifted_inp for _, out_shifted_inp in sorted(outputs_shifted_inp[fname])])
		
	#evaluate NMSE (and also other metrics) between shift(out) and out_shift_inp 
	
	METRIC_FUNCS = dict(
	NMSE=nmse,
	PSNR=psnr, 
	SSIM=ssim)

	metrics = Metrics(METRIC_FUNCS)
	print('Images reconstructed and gathered. Metrics (NMSE, PSNR, SSIM) calculation begins==>')
	for fname in tqdm(outputs.keys()):

		target = h5py.File(os.path.join(args.data, 'singlecoil_val', fname), 'r')
		# print(target.attrs['acquisition'])

		if args.evaluate_acquisition and evaluate_acquisition_names[args.evaluate_acquisition] != target.attrs['acquisition']:
			continue

		out = outputs[fname]
		out_shift_inp = outputs_shifted_inp[fname]
		shifted_out = np.roll(out, shift = shifts[fname], axis = (1, 2))

		metrics.push(target = shifted_out, recons = out_shift_inp)
	
	return metrics



def evaluate_out_of_dist_equivariance_metrics(dataloaders, model, args):
	
	evaluate_acquisition_names = {'PD': 'CORPD_FBK', 'PDFS': 'CORPDFS_FBK'}

	if args.eval_phase == 'validation':
		target_path = Path(os.path.join(args.data, 'singlecoil_val'))
	else:
		raise Exception('Other evaluation phases like train and test currently not supported.')
	

	outputs = defaultdict(list)
	outputs_shifted_inp = defaultdict(list)

	shifts = defaultdict(list)

	with torch.no_grad():
		with tqdm(dataloaders[args.eval_phase]) as pbar:
			for obs, gt,  mean, std, fname, slice_num, max_value in pbar:
				fname = fname[0]
				if len(obs.shape)==3:
					obs = obs.unsqueeze(1)
					gt = gt.unsqueeze(1)
	
				mean = mean.unsqueeze(1).unsqueeze(2)
				std = std.unsqueeze(1).unsqueeze(2)
				
				#construct shift for fname
				if fname in shifts.keys():
					shift = shifts[fname]
				else:
					shift = list(np.random.randint(low = -args.max_shift, high=args.max_shift, size=2))
					shifts[fname] = shift
				
				obs = obs.cuda()
				obs += args.eval_noise_level*torch.randn(obs.shape).cuda()
				reco = model(obs).cpu().clamp(-6, 6)
				reco_shift = model(torch.roll(obs, shifts = shift, dims = (2,3))).cpu().clamp(-6, 6)
				obs = obs.cpu()

				# undo the instance-normalized the output and target
				translated_reco = (reco * std + mean).detach().numpy().squeeze()
				translated_reco_shift = (reco_shift * std + mean).detach().numpy().squeeze()

				# collect slices into the volume it belongs to
				outputs[fname].append((slice_num.numpy(), translated_reco ))
				outputs_shifted_inp[fname].append((slice_num.numpy(), translated_reco_shift ))
				
	# numpy stack output images for individual fnames
	for fname in outputs.keys():
		outputs[fname] = np.stack([out for _, out in sorted(outputs[fname])])
		outputs_shifted_inp[fname] = np.stack([out_shifted_inp for _, out_shifted_inp in sorted(outputs_shifted_inp[fname])])
		
	#evaluate NMSE (and also other metrics) between shift(out) and out_shift_inp 
	
	METRIC_FUNCS = dict(
	NMSE=nmse,
	PSNR=psnr, 
	SSIM=ssim)

	metrics = Metrics(METRIC_FUNCS)
	print('Images reconstructed and gathered. Metrics (NMSE, PSNR, SSIM) calculation begins==>')
	for fname in tqdm(outputs.keys()):

		target = h5py.File(os.path.join(args.data, 'singlecoil_val', fname), 'r')
		# print(target.attrs['acquisition'])

		if args.evaluate_acquisition and evaluate_acquisition_names[args.evaluate_acquisition] != target.attrs['acquisition']:
			continue

		out = outputs[fname]
		out_shift_inp = outputs_shifted_inp[fname]
		shifted_out = np.roll(out, shift = shifts[fname], axis = (1, 2))

		metrics.push(target = shifted_out, recons = out_shift_inp)
	
	return metrics



def evaluate_out_of_dist_unshifted_metrics(dataloaders, model, args):
	
	evaluate_acquisition_names = {'PD': 'CORPD_FBK', 'PDFS': 'CORPDFS_FBK'}

	if args.eval_phase == 'validation':
		target_path = Path(os.path.join(args.data, 'singlecoil_val'))
	else:
		raise Exception('Other evaluation phases like train and test currently not supported.')
	

	outputs = defaultdict(list)
	shifts = defaultdict(list)
	with torch.no_grad():
		with tqdm(dataloaders[args.eval_phase]) as pbar:
			for obs, gt,  mean, std, fname, slice_num, max_value in pbar:
				fname = fname[0]
				if len(obs.shape)==3:
					obs = obs.unsqueeze(1)
					gt = gt.unsqueeze(1)
	
				mean = mean.unsqueeze(1).unsqueeze(2)
				std = std.unsqueeze(1).unsqueeze(2)
				
				#construct shift for fname
				if fname in shifts.keys():
					shift = shifts[fname]
				else:
					shift = list(np.random.randint(low = -args.max_shift, high=args.max_shift, size=2))
					shifts[fname] = shift

				obs = obs.cuda()
				obs += args.eval_noise_level*torch.randn(obs.shape).cuda()
				reco = model(obs).cpu().clamp(-6, 6)
				obs = obs.cpu()
				
					
				# undo the instance-normalized the output and target
				translated_reco = (reco * std + mean).detach().numpy().squeeze()

				# collect slices into the volume it belongs to
				outputs[fname].append((slice_num.numpy(), translated_reco ))
				
	# numpy stack output images for individual fnames
	for fname in outputs.keys():
		outputs[fname] = np.stack([out for _, out in sorted(outputs[fname])])
		
	#evaluate PSNR with original targets loaded from file
	
	METRIC_FUNCS = dict(
	NMSE=nmse,
	PSNR=psnr, 
	SSIM=ssim)

	metrics = Metrics(METRIC_FUNCS)
	print('Images reconstructed and gathered. Metrics (NMSE, PSNR, SSIM) calculation begins==>')
	for tgt_file in tqdm(target_path.iterdir()):
		
		fname = tgt_file.name
		target = h5py.File(tgt_file, 'r')
		recons = outputs[fname]
		
		if args.evaluate_acquisition and evaluate_acquisition_names[args.evaluate_acquisition] != target.attrs['acquisition']:
			continue

		target = target['reconstruction_esc'][()]
		target = fastmri_transforms.center_crop(target, (recons.shape[-1], recons.shape[-1]))
		target_shift = np.roll(target, shift = shifts[fname], axis = (1, 2))
		
		metrics.push(target_shift, recons)
		
	
	return metrics


# def evaluate_out_of_dist_noise_unshifted_metrics(dataloaders, model, args):
	
#     evaluate_acquisition_names = {'PD': 'CORPD_FBK', 'PDFS': 'CORPDFS_FBK'}

#     if args.eval_phase == 'validation':
#         target_path = Path(os.path.join(args.data, 'singlecoil_val'))
#     else:
#         raise Exception('Other evaluation phases like train and test currently not supported.')
	

#     outputs = defaultdict(list)

#     with torch.no_grad():
#         with tqdm(dataloaders[args.eval_phase]) as pbar:
#             for obs, gt,  mean, std, fname, slice_num, max_value in pbar:
#                 fname = fname[0]
#                 if len(obs.shape)==3:
#                     obs = obs.unsqueeze(1)
#                     gt = gt.unsqueeze(1)
	
#                 mean = mean.unsqueeze(1).unsqueeze(2)
#                 std = std.unsqueeze(1).unsqueeze(2)
				
#                 #construct shift for fname
#                 if fname in shifts.keys():
#                     shift = shifts[fname]
#                 else:
#                     shift = list(np.random.randint(low = -args.max_shift, high=args.max_shift, size=2))
#                     shifts[fname] = shift
				
#                 obs = obs.cuda()
#                 obs += args.eval_noise_level*torch.randn(obs.shape).cuda()
#                 reco = model(obs).cpu().clamp(-6, 6)
#                 reco_shift = model(torch.roll(obs, shifts = shift, dims = (2,3))).cpu().clamp(-6, 6)
#                 obs = obs.cpu()

#                 # undo the instance-normalized the output and target
#                 translated_reco = (reco * std + mean).detach().numpy().squeeze()
#                 translated_reco_shift = (reco_shift * std + mean).detach().numpy().squeeze()

#                 # collect slices into the volume it belongs to
#                 outputs[fname].append((slice_num.numpy(), translated_reco ))
#                 outputs_shifted_inp[fname].append((slice_num.numpy(), translated_reco_shift ))
				
#     # numpy stack output images for individual fnames
#     for fname in outputs.keys():
#         outputs[fname] = np.stack([out for _, out in sorted(outputs[fname])])
#         outputs_shifted_inp[fname] = np.stack([out_shifted_inp for _, out_shifted_inp in sorted(outputs_shifted_inp[fname])])
		
#     #evaluate NMSE (and also other metrics) between shift(out) and out_shift_inp 
	
#     METRIC_FUNCS = dict(
#     NMSE=nmse,
#     PSNR=psnr, 
#     SSIM=ssim)

#     metrics = Metrics(METRIC_FUNCS)
#     print('Images reconstructed and gathered. Metrics (NMSE, PSNR, SSIM) calculation begins==>')
#     for fname in tqdm(outputs.keys()):

#         target = h5py.File(os.path.join(args.data, 'singlecoil_val', fname), 'r')
#         # print(target.attrs['acquisition'])

#         if args.evaluate_acquisition and evaluate_acquisition_names[args.evaluate_acquisition] != target.attrs['acquisition']:
#             continue

#         out = outputs[fname]
#         out_shift_inp = outputs_shifted_inp[fname]
#         shifted_out = np.roll(out, shift = shifts[fname], axis = (1, 2))

#         metrics.push(target = shifted_out, recons = out_shift_inp)
	
#     return metrics





# def evaluate_psnr(val_loader, model, args):
# # function from ISTA-Unet fastmri code
#     outputs = defaultdict(list)
#     targets = defaultdict(list)

#     with torch.no_grad():
#         with tqdm(val_loader) as pbar:
#             for obs, gt,  mean, std, fname, slice_num, max_value  in pbar:
#                 obs = obs.unsqueeze(1)
#                 gt = gt.unsqueeze(1)

#                 mean = mean.unsqueeze(1).unsqueeze(2).unsqueeze(3)
#                 std = std.unsqueeze(1).unsqueeze(2).unsqueeze(3)

#                 reco = model(obs.cuda(args.gpu, non_blocking=True)).cpu()

#                 # undo the instance-normalized the output and target
#                 trans_obs = (obs * std + mean).detach().numpy().squeeze()
#                 trans_reco = (reco * std + mean).detach().numpy().squeeze()
#                 trans_target = (gt * std + mean).detach().numpy().squeeze()

	   

#                 for fname_idx in range(len(fname)):
#                     # collect slices into the volume it belongs to
#                     outputs[fname[fname_idx] ].append((slice_num.numpy()[fname_idx], trans_reco[fname_idx] ))
#                     targets[fname[fname_idx] ].append((slice_num.numpy()[fname_idx], trans_target[fname_idx] ))

#     print('Total number of volumes obtained from validation set: ', len(outputs))

#     stacked_outputs = defaultdict(list)
#     stacked_targets = defaultdict(list)

#     for fname in outputs.keys():

#         if args.evaluate_acquisition is not None:
#             f1 = h5py.File(args.data+'/singlecoil_val/'+fname, 'r') 

#             if not args.evaluate_acquisition+'_' in f1.attrs['acquisition']:
#                 continue
				
#         outputs_list = [out for _, out in sorted(outputs[fname], key=lambda x: x[0]) if out.shape == (320, 320) ]
#         targets_list = [target for _, target in sorted(targets[fname], key=lambda x: x[0]) if target.shape == (320, 320) ]
		
#         stacked_outputs[fname] = np.stack(outputs_list)
#         stacked_targets[fname] = np.stack(targets_list)

#     print('Number of volumes with the right fat-suppression mode to be evaluated on : ', len(stacked_outputs))

#     psnr_list = []
#     for fname in stacked_outputs.keys():
#         target = stacked_targets[fname]
#         recons = stacked_outputs[fname]

#         psnr_list.append( psnr(target, recons ) ) 

#     psnr_array = np.array( psnr_list  )

#     avg_psnr = np.mean(psnr_array)

	
#     # if args.evaluate:
#     #     f = open(args.out_dir+'/evaluate_psnr_result_final.txt', 'a')
#     #     f.write( '\n * PSNR {psnr1:.3f}'
#     #       .format(psnr1=avg_psnr) )
#     #     f.close()

#     #     pickle.dump(avg_psnr, open(args.out_dir+'/evaluate_psnr_result_final.p', 'wb'))

#     return avg_psnr



# def get_stacked_outputs(val_loader, model, shift, args):

#     outputs = defaultdict(list)
#     targets = defaultdict(list)

#     with torch.no_grad():
#         with tqdm(val_loader) as pbar:
#             for obs, gt,  mean, std, fname, slice_num, max_value  in pbar:
#                 obs = obs.unsqueeze(1)
#                 gt = gt.unsqueeze(1)

#                 mean = mean.unsqueeze(1).unsqueeze(2).unsqueeze(3)
#                 std = std.unsqueeze(1).unsqueeze(2).unsqueeze(3)

#                 gt = torch.roll(gt.cuda(args.gpu, non_blocking=True), dims = (2, 3), shifts = shift).cpu()
#                 reco = model(torch.roll(obs.cuda(args.gpu, non_blocking=True), dims = (2, 3), shifts = shift)).cpu()

#                 # undo the instance-normalized the output and target
#                 trans_obs = (obs * std + mean).detach().numpy().squeeze()
#                 trans_reco = (reco * std + mean).detach().numpy().squeeze()
#                 trans_target = (gt * std + mean).detach().numpy().squeeze()


#                 for fname_idx in range(len(fname)):
#                     # collect slices into the volume it belongs to
#                     outputs[fname[fname_idx] ].append((slice_num.numpy()[fname_idx], trans_reco[fname_idx] ))
#                     targets[fname[fname_idx] ].append((slice_num.numpy()[fname_idx], trans_target[fname_idx] ))

#     print('Total number of volumes obtained from validation set: ', len(outputs))

#     stacked_outputs = defaultdict(list)
#     stacked_targets = defaultdict(list)

#     for fname in outputs.keys():

#         if args.evaluate_acquisition is not None:
#             f1 = h5py.File(args.data+'/singlecoil_val/'+fname, 'r') 

#             if not args.evaluate_acquisition+'_' in f1.attrs['acquisition']:
#                 continue
			 

#         outputs_list = [out for _, out in sorted(outputs[fname], key=lambda x: x[0]) if out.shape == (320, 320) ]
#         stacked_outputs[fname] = np.stack(outputs_list)
		

#         targets_list = [target for _, target in sorted(targets[fname], key=lambda x: x[0]) if target.shape == (320, 320) ]
#         stacked_targets[fname] = np.stack(targets_list)


#     print('Number of volumes with the right fat-suppression mode to be evaluated on : ', len(stacked_outputs))

#     return stacked_outputs, stacked_targets





# def evaluate_psnr_adverserial(val_loader, model, args):
# # function from ISTA-Unet fastmri code
	
#     shift_options = [[i1, i2] for i2 in range(-1,2) for i1 in range(-2, 3) ]
#     shift_options.remove([0,0])
#     shift_options = [[0,0]]+shift_options

#     # shift_options = [[0,0]]

#     psnr_arrays_all_shifts = []


#     for k, shift in enumerate(shift_options):

#         print('\n==>Shift currently used: ', shift,'\n')

#         stacked_outputs, stacked_targets = get_stacked_outputs(val_loader, model, shift, args)
#         print('Stacked outputs obtained for shift: ', shift )

#         #PSNR calculation

#         print('==>PSNR calculation begins for shift: ', shift)

#         psnr_list = []
#         for fname in stacked_outputs.keys():
#             target = stacked_targets[fname]
#             recons = stacked_outputs[fname]

#             psnr_list.append( psnr(target, recons ) ) 
		
#         psnr_arrays_all_shifts.append(np.array( psnr_list  ))
	


#     psnr_arrays_all_shifts = np.array(psnr_arrays_all_shifts)

#     return psnr_arrays_all_shifts






# def evaluate_shift_psnr(val_loader, model, args):

#     outputs = defaultdict(list)
#     targets = defaultdict(list)

#     with torch.no_grad():
#         with tqdm(val_loader) as pbar:
#             for obs, gt,  mean, std, fname, slice_num, max_value  in pbar:
#                 obs = obs.unsqueeze(1).cuda(args.gpu, non_blocking=True)
#                 gt = gt.unsqueeze(1).cuda(args.gpu, non_blocking=True)

#                 shift = np.random.randint(-args.max_shift, args.max_shift, 2)

#                 obs = torch.roll(obs, shifts = (shift[0], shift[1]), dims = (2,3))
#                 gt = torch.roll(gt, shifts = (shift[0], shift[1]), dims = (2,3))

#                 mean = mean.unsqueeze(1).unsqueeze(2).unsqueeze(3)
#                 std = std.unsqueeze(1).unsqueeze(2).unsqueeze(3)

#                 reco = model(obs).detach().clamp(-6, 6).cpu()

#                 # undo the instance-normalized the output and target
#                 trans_obs = (obs.cpu() * std + mean).detach().numpy().squeeze()
#                 trans_reco = (reco * std + mean).detach().numpy().squeeze()
#                 trans_target = (gt.cpu() * std + mean).detach().numpy().squeeze()

#                 for fname_idx in range(len(fname)):
#                     # collect slices into the volume it belongs to
#                     outputs[fname[fname_idx] ].append((slice_num.numpy()[fname_idx], trans_reco[fname_idx] ))
#                     targets[fname[fname_idx] ].append((slice_num.numpy()[fname_idx], trans_target[fname_idx] ))

#     print('Total number of volumes obtained from validation set: ', len(outputs))
#     stacked_outputs = defaultdict(list)
#     stacked_targets = defaultdict(list)

#     for fname in outputs.keys():

#         if args.evaluate_acquisition is not None:
#             f1 = h5py.File(args.data+'/singlecoil_val/'+fname, 'r') 

#             if not args.evaluate_acquisition+'_' in f1.attrs['acquisition']:
#                 continue

				
#         outputs_list = [out for _, out in sorted(outputs[fname], key=lambda x: x[0]) if out.shape == (320, 320) ]
#         targets_list = [target for _, target in sorted(targets[fname], key=lambda x: x[0]) if target.shape == (320, 320) ]
		
#         stacked_outputs[fname] = np.stack(outputs_list)
#         stacked_targets[fname] = np.stack(targets_list)

#     print('Number of volumes with the right fat-suppression mode to be evaluated on : ', len(stacked_outputs))

#     psnr_list = []
#     for fname in stacked_outputs.keys():
#         target = stacked_targets[fname]
#         recons = stacked_outputs[fname]

#         psnr_list.append( psnr(target, recons ) ) 

#     psnr_array = np.array( psnr_list  )

#     avg_psnr = np.mean(psnr_array)

	
#     # if args.evaluate_shift:
#     #     f = open(args.out_dir+'/evaluate_shift_psnr_result_final.txt', 'a')
#     #     f.write( '\n * Shifted PSNR {psnr1:.3f}'
#     #       .format(psnr1=avg_psnr) )
#     #     f.close()

#         # pickle.dump(avg_psnr, open(args.out_dir+'/evaluate_shift_psnr_result_final.p', 'wb'))

#     return avg_psnr




def pickle_dump_and_write(object_to_write, object_to_pickle_save, path, file_name):

	pickle.dump(object_to_pickle_save, open( os.path.join(path, file_name + '.p') , 'wb'))

	f = open(os.path.join(path, file_name + '.txt'), 'a')
	f.write( str(object_to_write) )
	f.close()

	return 



def nmse(gt, pred):
	""" Compute Normalized Mean Squared Error (NMSE) """
	return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
	""" Compute Peak Signal to Noise Ratio metric (PSNR) """
	return peak_signal_noise_ratio(gt, pred, data_range=gt.max())


def ssim(gt, pred):
	""" Compute Structural Similarity Index Metric (SSIM). """
	return structural_similarity(
		gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range=gt.max()
	)



METRIC_FUNCS = dict(
	NMSE=nmse,
	PSNR=psnr,
	SSIM=ssim)

class Metrics:
	"""
	Maintains running statistics for a given collection of metrics.
	"""

	def __init__(self, metric_funcs):
		"""
		Args:
			metric_funcs (dict): A dict where the keys are metric names and the
				values are Python functions for evaluating that metric.
		"""
		self.metrics = {metric: Statistics() for metric in metric_funcs}

	def push(self, target, recons):
		for metric, func in METRIC_FUNCS.items():
			self.metrics[metric].push(func(target, recons))

	def means(self):
		return {metric: stat.mean() for metric, stat in self.metrics.items()}

	def stddevs(self):
		return {metric: stat.stddev() for metric, stat in self.metrics.items()}

	def __repr__(self):
		means = self.means()
		stddevs = self.stddevs()
		metric_names = sorted(list(means))
		return " ".join(
			f"{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}"
			for name in metric_names )






def save_metrics(metrics, args, evaluation_type):

	metrics_dict = {}
	for key in metrics.metrics.keys():
		
		individual_dict = {'mean': metrics.metrics[key].mean(), 'stddev': metrics.metrics[key].stddev()}
		metrics_dict[key] = individual_dict


	if 'out_of_dist' in evaluation_type:
		save_path = os.path.join(args.out_dir, 'out_of_dist_exps')
		if not os.path.isdir(save_path):
			os.mkdir(save_path)
	else:
		save_path = args.out_dir

	pickle.dump(metrics_dict, open(os.path.join(save_path,'evaluate_metrics_'+evaluation_type+'.p'), 'wb'))

	f = open(os.path.join(save_path,'evaluate_metrics_'+evaluation_type+'.txt'), 'a')
	f.write( str(metrics) )
	f.close()




def validate_save(val_loader, mean, std, args):
	import matplotlib.pyplot as plt
	import os
	for i, (input, target) in enumerate(val_loader):
		img = (255*np.clip(input[0,...].data.cpu().numpy()*np.array(std)[:,None,None] + mean[:,None,None],0,1)).astype('uint8').transpose((1,2,0))
		plt.imsave(os.path.join(args.out_dir,'%05d.png'%i),img)

# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
def save_checkpoint(state, is_best, epoch, out_dir='./'):
	torch.save(state, os.path.join(out_dir,'checkpoint.pth.tar'))
	if(epoch % 10 == 0):
		torch.save(state, os.path.join(out_dir,'checkpoint_%03d.pth.tar'%epoch))
	if is_best:
		shutil.copyfile(os.path.join(out_dir,'checkpoint.pth.tar'), os.path.join(out_dir,'model_best.pth.tar'))


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


# def adjust_learning_rate(optimizer, epoch, args):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.lr * (0.1 ** (epoch // args.lr_step))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr




# def psnr_calc(output, target, mean, std):

#     # undo the instance-normalized the output and target
#     output = (output * std + mean).detach().squeeze(1)
#     target = (target * std + mean).detach().squeeze(1)

#     error = output - target

#     abs_peak_vals = torch.max(target.abs().reshape(B, -1), dim = 1)

#     error_power = torch.norm(error, dim = (1, 2))**2/np.prod(error.shape[1:3])

#     psnr = 10*torch.log10(abs_peak_vals/error_power)

#     return [torch.mean(psnr).item()]








if __name__ == '__main__':
	main()

import argparse
import os
import random
import shutil
import time
import warnings
import sys
import numpy as np
import os

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

from runstats import Statistics



import pickle
import sys
sys.path.insert(1, '../')
sys.path.insert(1, '../../')

import models
from models import baseline_unet
from models.aps_models import unet_aps
from models.lpf_models import unet_lpf

from utils.load_lodopab_ct import get_dataloaders_ct

from skimage.metrics import structural_similarity, peak_signal_noise_ratio


parser = argparse.ArgumentParser(description='PyTorch CT reconstruction')



parser.add_argument('--out-dir', dest='out_dir', default='/raid/anadi/shift_equivariant_net_results/ct_exps_results/', type=str,
				help='output directory')
													
parser.add_argument('--data', metavar='DIR', default='/raid/datasets/ct_lodopab/',
					help='path to dataset')
					  
#..................training and evaluation args............
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
					help='number of data loading workers (default: 4)')
parser.add_argument('-ep', '--epochs', default=20, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
					metavar='N',
					help='mini-batch size, this is the total '
						 'batch size of all GPUs on the current node when '
						 'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--criterion', default='nn.MSELoss', type=str,
				help='Loss function to be optimized over. ')

parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
					metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_step', default=5, type=float,
					help='number of epochs before stepping down learning rate')

parser.add_argument('--lr_decay_factor', default=0.1, type=float,
					help='Multiplicative decay factor when stepping down lr.')

parser.add_argument('--dataset_image_center_crop', default=352, type=int,
					help='Center crop size of images.')


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


parser.add_argument('--evaluate', action='store_true',
					help='Evaluate models on dataset with PSNR and SSIM metric.')

parser.add_argument('--evaluate-equivariance-metrics', dest = 'evaluate_equivariance_metrics', action='store_true',
					help='Evaluate equivariance metrics between shifted output and output of shifted input.')



parser.add_argument('--evaluate-diff-psnr-shift', dest = 'evaluate_diff_psnr_shift', action='store_true',
					help='Evaluate the max decline in PSNR ccaused by shifts.')


parser.add_argument('--num_shifts_diff_psnr', default = 10, type = int,
					help='Number of random shifts to check for PSNR decline.')


parser.add_argument('--max_shift', default = 16, type = int,
					help='Maximum possible shift to use during data augmentation or shift evaluation.')


parser.add_argument('--evaluate_data_phase', default = 'test',
					help='train/validation/test set to evaluate over.')





#.................  misc args ................................
parser.add_argument('-ba', '--batch-accum', default=1, type=int,
					metavar='N',
					help='number of mini-batches to accumulate gradient over before updating (default: 1)')
parser.add_argument('--embed', dest='embed', action='store_true',
					help='embed statement before anything is evaluated (for debugging)')
parser.add_argument('--val-debug', dest='val_debug', action='store_true',
					help='debug by evaluating on val set')

parser.add_argument('--val-train-debug', dest='val_train_debug', action='store_true',
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



highest_psnr = 0.0
lowest_loss = 1e6


def main():

	args = parser.parse_args()
		
	if args.visible_devices !='all':
		print('Setting GPUs ', args.visible_devices)

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


	dataset_dict = {'batch_size': args.batch_size, 'num_workers': args.workers, 'include_validation': True, 
	'cache_dir': os.path.join(args.data, 'cache_fbp_lodopab'), 'dataset_image_center_crop': args.dataset_image_center_crop}

	dataloaders = get_dataloaders_ct(**dataset_dict)

	
	if args.evaluate:

		args.batch_size = 1
		model.eval()
		print('Evaluating model on '+args.evaluate_data_phase+' set.==>')
		metrics = evaluate_metrics(dataloaders[args.evaluate_data_phase], model, args)
		print(metrics)

		save_metrics(metrics, args, file_name = 'evaluate_metrics_'+args.evaluate_data_phase+'_images')

		return 

	if args.evaluate_equivariance_metrics:

		args.batch_size = 1
		model.eval()
		print('Evaluating equivariance metrics on '+args.evaluate_data_phase+' set.==>')

		# fix random seed to use same shifts for each eval
		np.random.seed(0)
		random.seed(0)
		torch.manual_seed(0)

		equivariance_metrics = evaluate_equivariance_metrics(dataloaders[args.evaluate_data_phase], model, args)
		print(equivariance_metrics)

		save_metrics(equivariance_metrics, args, file_name = 'evaluate_equivariance_metrics_'+args.evaluate_data_phase+'images')
		return


	if args.evaluate_diff_psnr_shift:

		args.batch_size = 1
		model.eval()
		print('Evaluating PSNR decline with shifts on '+args.evaluate_data_phase+' set.==>')

		# fix random seed to use same shifts for each eval
		np.random.seed(0)
		random.seed(0)
		torch.manual_seed(0)

		max_diff_psnr_list = evaluate_diff_psnr(dataloaders[args.evaluate_data_phase], model, args)

		path = args.out_dir
		file_name = 'eval_diff_psnr_shift_'+args.evaluate_data_phase+'_images_shift_'+str(args.num_shifts_diff_psnr)

		eval_string = 'Mean of max abs difference between psnr_unshift and psnr_shift: ' + str(np.mean(max_diff_psnr_list))+'\n'
		eval_string += 'Max of max abs difference between psnr_unshift and psnr_shift: ' + str(np.max(max_diff_psnr_list))+'\n\n'
 
 
		pickle_dump_and_write(eval_string, max_diff_psnr_list, path, file_name)
		return 



	if args.val_debug:  #run the  evaluation on validation set
		
		# model.eval()
		loss = validate_debug(dataloaders['train'], model, criterion, args)
		return 


	if(args.val_train_debug): # debug mode - train on val set for faster epochs
		train_loader = val_loader

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

	training_loss_epoch_list = []
	validation_loss_epoch_list = []

	print('Training begins==>')
	for epoch in range(args.start_epoch, args.epochs):
		if args.distributed:
			train_sampler.set_epoch(epoch)

		

		if(args.wandb):
			wandb.log({'learning_rate': optimizer.param_groups[0]['lr']},
					  commit=False)

		# train for one epoch
		train_loss = train(dataloaders['train'], model, criterion, optimizer, epoch, args)
		training_loss_epoch_list.append(train_loss)

		scheduler.step()

		# evaluate on validation set
		val_loss = validate(dataloaders['validation'], model, criterion, args)
		validation_loss_epoch_list.append(val_loss)

		# remember best scenario (lowest loss) and save checkpoint
		is_best = val_loss < lowest_loss
		lowest_loss = min(lowest_loss, val_loss)

		if not args.multiprocessing_distributed or (args.multiprocessing_distributed
				and args.rank % ngpus_per_node == 0):
			save_checkpoint({
				'epoch': epoch + 1,
				'arch': args.arch,
				'state_dict': model.state_dict(),
				'lowest_loss': lowest_loss,
				'optimizer' : optimizer.state_dict(),
			}, is_best, epoch, out_dir=args.out_dir)

			np.save(os.path.join(args.out_dir, 'train_loss_list_node_zero.npy'), training_loss_epoch_list)
			np.save(os.path.join(args.out_dir, 'validation_loss_epoch_list.npy'), validation_loss_epoch_list)








def train(data_loader, model, criterion, optimizer, epoch, args):

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
	for i, (x, d) in enumerate(data_loader):
		# measure data loading time
		data_time.update(time.time() - end)
		
		# if args.gpu is not None:
		x = x.cuda(args.gpu, non_blocking=True)
		d = d.cuda(args.gpu, non_blocking=True)
		
		if args.circular_data_aug:
			shift = np.random.randint(-args.max_shift, args.max_shift,size=2)
			x = torch.roll(x, shifts = (shift[0], shift[1]) , dims = (2, 3) )
			d = torch.roll(d, shifts = (shift[0], shift[1]) , dims = (2, 3) )



		# compute output
		output = model(x)
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
				   epoch, i, len(data_loader), batch_time=batch_time,
				   data_time=data_time, loss=losses))

			string = '\n\nEpoch: [{0}][{1}/{2}]\t''Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t''Data {data_time.val:.3f} ({data_time.avg:.3f})\t''Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(data_loader), batch_time=batch_time, data_time=data_time, loss=losses)
				  

			f = open(os.path.join(args.out_dir,'train_log.txt'), 'a')
			f.write( string )
			f.close()

			if(args.wandb):
				import wandb
				global_step = i + (epoch * len(data_loader))
				wandb.log(
					{
						'train_loss': losses.val,
						'train_avg_loss': losses.avg,
						'epoch': 1.*global_step/len(data_loader), 
					},
					step=global_step)

		if(i > args.max_train_iters):
			break


	return losses.avg


def validate(data_loader, model, criterion, args):

	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	# psnr1 = AverageMeter()

	# switch to train mode
	model.eval()

	loss = 0

	end = time.time()
	with torch.no_grad():
		for i, (x, d) in enumerate(data_loader):
			# measure data loading time
			data_time.update(time.time() - end)
			
			# if args.gpu is not None:
			x = x.cuda(args.gpu, non_blocking=True)
			d = d.cuda(args.gpu, non_blocking=True)
			

			# compute output
			output = model(x)
			loss = criterion(output, d)

			losses.update(loss.item(), x.size(0))

			
			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.print_freq == 0:
				print('Test: [{0}/{1}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
					   i, len(data_loader), batch_time=batch_time,
					   data_time=data_time, loss=losses))

				string = '\nTest: [{0}/{1}]\t''Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t''Data {data_time.val:.3f} ({data_time.avg:.3f})\t''Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(data_loader), batch_time=batch_time, data_time=data_time, loss=losses)
					  

				f = open(os.path.join(args.out_dir,'val_log.txt'), 'a')
				f.write( string )
				f.close()

				if(args.wandb):
					import wandb
					global_step = i + (epoch * len(data_loader))
					wandb.log(
						{
							'val_loss': losses.val,
							'val_avg_loss': losses.avg,
							'epoch': 1.*global_step/len(data_loader), 
						},
						step=global_step)




	return losses.avg


def validate_debug(data_loader, model, criterion, args):

	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	# psnr1 = AverageMeter()

	# switch to train mode
	model.eval()

	loss = 0

	end = time.time()
	with torch.no_grad():
		for i, (x, d) in enumerate(data_loader):
			# measure data loading time
			data_time.update(time.time() - end)
			
			# if args.gpu is not None:
			x = x.cuda(args.gpu, non_blocking=True)
			d = d.cuda(args.gpu, non_blocking=True)

			# compute output
			output = model(x)

			print('Max-min stats of input: ', x.max(), x.min())
			print('Max-min stats of grnd truth: ', d.max(), d.min())
			print('Max-min stats of output: ', output.max(), output.min())
			print('Max-min stats of error b/w out and grnd truth: ', (output - d).abs().max(), (output - d).abs().min())
			print('\n\n')

			if i==3:
				break
			loss = criterion(output, d)

			losses.update(loss.item(), x.size(0))

			
			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.print_freq == 0:
				print('Test: [{0}/{1}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
					   i, len(data_loader), batch_time=batch_time,
					   data_time=data_time, loss=losses))

				string = '\nTest: [{0}/{1}]\t''Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t''Data {data_time.val:.3f} ({data_time.avg:.3f})\t''Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(data_loader), batch_time=batch_time, data_time=data_time, loss=losses)
					  

				f = open(os.path.join(args.out_dir,'val_log.txt'), 'a')
				f.write( string )
				f.close()

				if(args.wandb):
					import wandb
					global_step = i + (epoch * len(data_loader))
					wandb.log(
						{
							'val_loss': losses.val,
							'val_avg_loss': losses.avg,
							'epoch': 1.*global_step/len(data_loader), 
						},
						step=global_step)




	return losses.avg




def evaluate_metrics(dataloader, model, args):

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

				out = model(obs.cuda()).cpu()

				metrics.push(gt[0, 0, :, :].numpy(), out[0, 0, :, :].numpy())

	return metrics



def evaluate_equivariance_metrics(dataloader, model, args):

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

				shift = list(np.random.randint(low = -args.max_shift, high=args.max_shift, size=2))

				obs = obs.cuda()
				shifted_obs = torch.roll(obs, shifts = shift, dims = (2,3))

				out = model(obs)
				out_shifted_inp = model(shifted_obs)

				shifted_out = torch.roll(out, shifts = shift, dims = (2,3))


				metrics.push(shifted_out[0, 0, :, :].detach().cpu().numpy(), out_shifted_inp[0, 0, :, :].detach().cpu().numpy())

	return metrics



def evaluate_diff_psnr(dataloader, model, args):

	if args.batch_size!=1:
		raise Exception('Batch size not 1 for PSNR and SSIM evaluation.')

	max_diff_psnr_list = []

	model.eval()
	with torch.no_grad():
		with tqdm(dataloader) as pbar:
			for obs, gt in pbar:
				
				obs = obs.cuda()

				out = model(obs).cpu().numpy()
				psnr_unshifted = psnr(gt[0, 0, :, :], out[0, 0, :, :])

				diff_psnr_curr_list = []
				for i1 in range(args.num_shifts_diff_psnr):
					shift = list(np.random.randint(low = -args.max_shift, high=args.max_shift, size=2))


					out_shifted_inp = model(torch.roll( obs, shifts = shift, dims = (2, 3) )).cpu().numpy()
					shifted_target = torch.roll( gt, shifts = shift, dims = (2, 3) )

					psnr_shift = psnr(shifted_target[0, 0, :, :], out_shifted_inp[0, 0, :, :])

					diff_psnr_curr_list.append(np.abs(psnr_unshifted - psnr_shift))

				max_diff_psnr_list.append(np.max(diff_psnr_curr_list))

	return max_diff_psnr_list

	






















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
	return peak_signal_noise_ratio(gt, pred, data_range=gt.max() - gt.min())


def ssim(gt, pred):
	""" Compute Structural Similarity Index Metric (SSIM). """
	return structural_similarity(
		gt, pred, multichannel=True, data_range=gt.max()
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


def save_metrics(metrics, args, file_name):

	metrics_dict = {}
	for key in metrics.metrics.keys():
		
		individual_dict = {'mean': metrics.metrics[key].mean(), 'stddev': metrics.metrics[key].stddev()}
		metrics_dict[key] = individual_dict

	save_path = args.out_dir

	pickle.dump(metrics_dict, open(os.path.join(save_path,file_name+'.p'), 'wb'))

	f = open(os.path.join(save_path, file_name + '.txt'), 'a')
	f.write( str(metrics)+'\n\n' )
	f.close()


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




if __name__ == '__main__':
	main()

























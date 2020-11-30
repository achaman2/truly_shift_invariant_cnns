# %%
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import sys
import os
import argparse
import random
import pickle

base_path = '../'
sys.path.insert(1,'../')
print(base_path+'shift_invariant_nets/')



import model_classes.models_for_cifar10

from utils.model_functions import compute_num_params
from utils.plot_functions import plot_grid_torch_figs, plot_save_train_val_list
from utils.file_functions import create_folders, create_folders_direct_path
from utils.for_circular_pad_exps.get_dataloaders_circular import get_dataloaders_circular
from utils.for_circular_pad_exps.train_validate_circular import train_and_validate_circular_all_epochs, evaluate_circular, evaluate_circular_flip, evaluate_circular_random_erase



# %%


# %%
def set_random_seeds(seed):
    
    if seed is not None:
    
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        np.random.seed(seed)
        random.seed(seed)
        
#         torch.backends.cudnn.benchmark = False
        
    else:
        return

    
def _init_fn(worker_id):
    np.random.seed(0)
    

# %%
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

# cifar10_path = base_path+'shift_invariant_nets_before_sept2020/data'

cifar10_path = '/raid/datasets/cifar-10'


parser.add_argument('--results_base_path', default='/raid/anadi/shift_invariant_nets_results/cifar10_circularpad_exps/results/', 
                    type=str, help='directory name where results to be stored')

parser.add_argument('--model_folder', default='', 
                    type=str, help='directory within results where each exp/model is to be stored')

parser.add_argument('--device_id', default='0', 
                    type=str, help='CUDA Id for GPU/use cpu if if is cpu')

# random seed flags
parser.add_argument('--seed_num', default=None, 
                    help='random seed')

parser.add_argument('--cudnn_deterministic', action = 'store_true',
                    help='random seed')



# resuming operation
parser.add_argument('--resume', action = 'store_true',
                    help='flag if model has to be loaded from model_folder directory')



# dataset and dataloader params
parser.add_argument( '--dataset', default='cifar10', 
                    help='dataset to train the model on' )

parser.add_argument( '--dataset_path', default = cifar10_path, 
                    help='dataset path' )

# parser.add_argument( '--image_pad_len', default=-1, type = int,  
#                     help='amount of zero padding to be done to the input (this overrides the padding amt decided by max_shift). Only used if >0' )

parser.add_argument('--batch_size', default=256, 
                    type=int, help='learning rate scheduler step size')

parser.add_argument('--base_center_crop', default=32, 
                    type=int, help='part of the center crop length without zeros')

parser.add_argument('--data_augmentation_flag', action = 'store_true',
                    help='flag to set data augmentation on/off')

parser.add_argument('--train_split', default=0.9, 
                    help='Fraction of training set to be used for training, while the rest used for validation')

parser.add_argument('--pin_memory', default=True, 
                    help='loads data to gpu in pinned memory (useful to speed things up)')

parser.add_argument('--num_workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument( '--max_shift', default=3, type=int,
                    help='highest shift used for consistency evaluation and data augmentation (if needed)')


parser.add_argument( '--validate_consistency', action = 'store_true',
                    help='Flag used to perform consistency check in validation stage as well')



# model params
parser.add_argument('--pretrained', action='store_true',
                    help='use pre-trained model')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet20', 
                    help='model architecture')

parser.add_argument('--conv_pad_type', default='circular', 
                    help='padding type used inside the conv net')

parser.add_argument('--filter_size', default=1, type = int, 
                    help='filter size used in lpf models')

parser.add_argument('--num_layers_scale', default=1, type = int, 
                    help='Multiplier for the number of channels used in resnet layers. 1 keeps the number to be the same as that from resnet paper')


# optimizer params
parser.add_argument('--lr', default=0.1, 
                    type=float, help='learning rate')

parser.add_argument('--momentum', default=0.9, 
                    type=float, metavar='M', help='momentum')

parser.add_argument('--weight_decay', default=5e-4, type = float,
                    help='Weight decay for model params in optimizer')

parser.add_argument('--step_size_lr', default=100, 
                    type=int, help='learning rate scheduler step size')

parser.add_argument('--scheduler_milestones', default=[100, 150], 
                    type=list, help='learning rate scheduler step size')

parser.add_argument('--scheduler_type', default='StepLR', 
                     help='learning rate scheduler step size')


parser.add_argument('--gamma', default=0.1, 
                    type=float, help='learning rate decay factor')

parser.add_argument('--scheduler_flag', default=True, 
                    help='True if learning rate decay is needed to be performed')

parser.add_argument('--num_epochs', default=250, 
                    type=int, help='number of epochs')

parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('--evaluate_test', action = 'store_true',
                    help='If the model has to be evaluated on the test set')

parser.add_argument('--evaluate_only', action = 'store_true',
                    help='If the model has to be evaluated on the test set and no training is needed. To be used if model has been trained a priori.')

parser.add_argument('--evaluate_on_flips', action = 'store_true',
                    help='Model will be evaluated on the test, but with images of the set flipped.')

parser.add_argument('--evaluate_on_erase', action = 'store_true',
                    help='Model will be evaluated on the test, but with images having randomly erased patches.')

parser.add_argument('--random_erase_patch', default = 3, type = int,
                    help='For evaluate_on_erase size of the patch to be erased')


parser.add_argument('--apspool_criterion', default = 'l2',
                    help='Type of criterion used for selecting poly component in APS')



def main():

    args = parser.parse_args()
    
    results_base_path = args.results_base_path
    model_folder = args.model_folder
    model_folder_path = results_base_path + model_folder
    
    DEVICE_ID = args.device_id
    os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_ID
    
    torch.backends.cudnn.benchmark = True
    
    # dataset and data loadersparams

    BATCH_SIZE = args.batch_size
    DATASET = args.dataset
    DATASET_PATH = args.dataset_path
    TRAIN_SPLIT = args.train_split

    BASE_CENTER_CROP = args.base_center_crop
    DATA_AUGMENTATION_FLAG = args.data_augmentation_flag
    PIN_MEMORY = args.pin_memory
    NUM_WORKERS = args.num_workers
    
    
    # model params
    PRE_TRAINED = args.pretrained
    ARCH = args.arch
    CONV_PAD_TYPE = args.conv_pad_type
    FILTER_SIZE = args.filter_size
    NUM_LAYERS_SCALE = args.num_layers_scale
    APSPOOL_CRITERION = args.apspool_criterion
    
    if ARCH[0:8] == 'resnet20' or ARCH[0:8] == 'resnet56':
        LAYER_CHANNELS = [16*NUM_LAYERS_SCALE, 32*NUM_LAYERS_SCALE, 64*NUM_LAYERS_SCALE]
    
    else:
        LAYER_CHANNELS = [64, 128, 256, 512]
    
    if CONV_PAD_TYPE !='circular':
        raise ValueError('This script can only used for circular pads')
    
    # seed params and flags
    
    SEED_NUM = eval(args.seed_num)
    CUDNN_DETERMINISTIC = args.cudnn_deterministic
    
    
    # optimizer params

    LEARNING_RATE = args.lr
    MOMENTUM = args.momentum
    WEIGHT_DECAY = args.weight_decay
    NUM_EPOCHS = args.num_epochs
    START_EPOCH = args.start_epoch
    RESUME = args.resume
    
    SCHEDULER_FLAG = args.scheduler_flag
    STEP_SIZE_LR = args.step_size_lr
    GAMMA = args.gamma
    SCHEDULER_TYPE = args.scheduler_type
    SCHEDULER_MILESTONES = args.scheduler_milestones
    VALIDATE_CONSISTENCY = args.validate_consistency
    MAX_SHIFT = args.max_shift
    
    DATA_LOADER_SEED = 0 #always seed data loader the same way
    EVAL_SEED = 0
    
    if SEED_NUM is not None:

        set_random_seeds(SEED_NUM)
        WORKER_INIT_FUNCTION = _init_fn
        
    else:
        WORKER_INIT_FUNCTION = None
        
    
    if CUDNN_DETERMINISTIC == True:
        torch.backends.cudnn.deterministic = True
    
    
    print('Architecture used: ', ARCH)
    print('Number of channels in inner layers: ', LAYER_CHANNELS)
    print('Filter size: ', FILTER_SIZE)
    
    
    #     Store all flags in respective dicts

    dataset_dict = {'batch_size': BATCH_SIZE, 
                    'dataset': DATASET, 
                    'dataset_path':DATASET_PATH,
                    'train_split': TRAIN_SPLIT,
                    'base_center_crop': BASE_CENTER_CROP,
                   'pin_memory': PIN_MEMORY, 
                    'num_workers': NUM_WORKERS, 
                    'worker_init_fn': WORKER_INIT_FUNCTION,
                    'data_loader_seed': DATA_LOADER_SEED,
                   }
    
    model_dict = {'conv_pad_type': CONV_PAD_TYPE,
                  'dataset_to_train': DATASET,
                  'pretrained': PRE_TRAINED,
                 'filter_size': FILTER_SIZE,
                 'layer_channels':LAYER_CHANNELS}
                 

    model_dict1 = {'arch': ARCH, **model_dict}


    randomness_params_dict = {'seed_num': SEED_NUM, 'worker_init_fn': WORKER_INIT_FUNCTION,
                        'num_workers':NUM_WORKERS, 'data_loader_seed': DATA_LOADER_SEED}
    
    optimizer_dict = {'lr': LEARNING_RATE,
                     'momentum': MOMENTUM,
                     'weight_decay': WEIGHT_DECAY,
                     }

    misc_dict = {'resume': RESUME, 'scheduler_flag': SCHEDULER_FLAG, 'scheduler_type': SCHEDULER_TYPE,
                                         'scheduler_milestones':SCHEDULER_MILESTONES,
                 'max_shift': MAX_SHIFT
                                        }

    
    if str.endswith(ARCH, 'lpf'):
        from model_classes.models_for_cifar10.lpf_models.resnet import resnet18_lpf, resnet34_lpf, resnet50_lpf, resnet20_lpf, resnet56_lpf

    elif str.endswith(ARCH, 'aps'):
        from model_classes.models_for_cifar10.aps_models.resnet import resnet18_aps, resnet34_aps, resnet50_aps, resnet20_aps, resnet56_aps
        
        model_dict['apspool_criterion'] = APSPOOL_CRITERION
        
        
        
#         from resnet import resnet20_aps, resnet56_aps

    else:
        from model_classes.models_for_cifar10.vanila_models.resnet import resnet18, resnet34, resnet50, resnet20, resnet56

    

    

    # ##################################   BEGIN  ##################################

    print(model_folder_path)
    if not os.path.isdir(model_folder_path):

        currentDirectory = os.getcwd()
        create_folders_direct_path(model_folder_path)
        print('Directory created')

    else:
        print('Directory exists')


    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available() and DEVICE_ID!='cpu':
        device = 'cuda'
    else:
        device = torch.device('cpu')


# ............................................    get dataloaders  .....................................
    set_random_seeds(DATA_LOADER_SEED)
    data_loaders = get_dataloaders_circular(**dataset_dict)

    # ...........................................    initialize model  ...........................................    
    
    set_random_seeds(SEED_NUM)
    model = eval(ARCH)(**model_dict)
    model.to(device)
    compute_num_params(model)
    print('Number of classes: ', model.fc.weight.shape[0])
    print()
    
    
    # ...........................................     initialize optimizer   ...........................................    

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), **optimizer_dict)


    if SCHEDULER_FLAG == True:

        if SCHEDULER_TYPE == 'StepLR':

            scheduler_dict = {
                     'step_size': STEP_SIZE_LR,
                      'gamma': GAMMA}

            scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_dict, last_epoch=args.start_epoch - 1)

        elif SCHEDULER_TYPE == 'MultiStepLR':

            scheduler_dict = {
                     'milestones': SCHEDULER_MILESTONES, 'gamma': GAMMA}

            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_dict, last_epoch=args.start_epoch - 1)

    else:
        scheduler = None

    best_accuracy = 0

    
    #...........................................    RESUME FROM CHECKPOINT IF ASKED FOR ...............................

    if RESUME:
        print('==> Resuming from checkpoint (model with best validation performance)...')
        checkpoint = torch.load(model_folder_path + '/models/model_and_optim_best_checkpoint.pt')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        best_accuracy = checkpoint['accuracy']
        START_EPOCH = checkpoint['epoch']


        
#    ...........................................     Evaluate models if flags ask for it ...........................................    


    
    if args.evaluate_on_erase:
        
        print('==> Evaluating accuracy and consistency on randomly erased test set images')
        checkpoint = torch.load(model_folder_path + '/models/model_and_optim_best_checkpoint.pt')
        model.load_state_dict(checkpoint['model'])
        model.eval()
        
        set_random_seeds(EVAL_SEED)
        test_accuracy_top1, test_total_consistency, test_string_status = evaluate_circular_random_erase(model, criterion, data_loaders['test'], max_shift = MAX_SHIFT, device = device, random_erase_patch = args.random_erase_patch, phase = 'test')
            
        
        string = 'Test results evaluated for random erase patch size: '+str(args.random_erase_patch)+'\n'
        string+=test_string_status
        
        erased_test_set_results  = {'erased_accuracy': test_accuracy_top1, 'erased_consistency': test_total_consistency,
                                   'random_erase_size': args.random_erase_patch}
        pickle.dump( erased_test_set_results, open(model_folder_path+'/erased_patch'+str(args.random_erase_patch)+'_test_set_results.p', "wb" ) )
        f = open(model_folder_path+'/erased_test_set_results.txt', 'a')
        f.write( string )
        f.close()
        
        if os.path.isdir('./results/'+model_folder):
            f = open('./results/'+model_folder+'/erased_test_set_results.txt', 'a')
            f.write( string )
            f.close()
            pickle.dump( erased_test_set_results, open('./results/'+model_folder+'/erased_patch'+str(args.random_erase_patch)+'_test_set_results.p', "wb" ) )
        
        return 
               


    if args.evaluate_on_flips:
        
        print('==> Evaluating accuracy and consistency on flipped test set')
        checkpoint = torch.load(model_folder_path + '/models/model_and_optim_best_checkpoint.pt')
        model.load_state_dict(checkpoint['model'])
        model.eval()
        set_random_seeds(EVAL_SEED)
        test_accuracy_top1, test_total_consistency, test_string_status = evaluate_circular_flip(model, criterion, data_loaders['test'], max_shift = MAX_SHIFT, device = device, phase = 'test')
        
        flipped_test_set_results  = {'flipped_accuracy': test_accuracy_top1, 'flipped_consistency': test_total_consistency }
        f = open(model_folder_path+'/flipped_test_set_results.txt', 'a')
        f.write( test_string_status )
        f.close()
        pickle.dump( flipped_test_set_results, open(model_folder_path+'/flipped_test_set_results.p', "wb" ) )
        
        
        if os.path.isdir('./results/'+model_folder):
            f = open('./results/'+model_folder+'/flipped_test_set_results.txt', 'a')
            f.write( test_string_status )
            f.close()
            pickle.dump( flipped_test_set_results, open('./results/'+model_folder+'/flipped_test_set_results.p', "wb" ) )
        
        return 
        
    if args.evaluate_only:
        print('==> Evaluating on Test set')
        
        if os.path.isfile(model_folder_path + '/models/model_and_optim_best_checkpoint.pt'):
            checkpoint = torch.load(model_folder_path + '/models/model_and_optim_best_checkpoint.pt')
            model.load_state_dict(checkpoint['model'])
            model.eval()
        
        else:
            print('No checkpoint found. Evaluating on untrained model.')
        
        set_random_seeds(EVAL_SEED)
        
        test_accuracy_top1, test_total_consistency, test_string_status = evaluate_circular(model, criterion, data_loaders['test'], max_shift = MAX_SHIFT, device = device, phase = 'test')

        test_set_results = {'accuracy': test_accuracy_top1, 'consistency': test_total_consistency }

        f = open(model_folder_path+'/test_set_results.txt', 'a')
        f.write( test_string_status )
        f.close()
        pickle.dump( test_set_results, open('./results/'+model_folder+'/test_set_results.p', "wb" ) )

        return 
    

#     ...........................................    construct all remaining dicts and save everything in config dict ...........................................    
    
    training_dict = {'scheduler_flag': SCHEDULER_FLAG, 'start_epoch': START_EPOCH, 'num_epochs': NUM_EPOCHS,
                     'model_folder': model_folder, 'best_accuracy': best_accuracy,
                    'data_augmentation_flag': DATA_AUGMENTATION_FLAG,
                    'validate_consistency': VALIDATE_CONSISTENCY,
                    'max_shift': MAX_SHIFT}

    config_dict = {**dataset_dict, **model_dict1, **optimizer_dict, **scheduler_dict, 
                   **misc_dict, **training_dict, **randomness_params_dict}


    f = open(model_folder_path+'/config_dict.txt', 'w')
    f.write( str(config_dict) )
    f.close()


    pickle.dump( config_dict, open(model_folder_path+'/config_dict.p', "wb" ) )


    #...........................................     TRAINING BEGINS ...........................................    

    print('==> Training begins')
    set_random_seeds(SEED_NUM)
    training_result = train_and_validate_circular_all_epochs(model, criterion, data_loaders, optimizer, scheduler, 
                                                    **training_dict, model_folder_path = model_folder_path, device = device)


        #...........................................     EVALUATION BEGINS ...........................................    

#     if args.evaluate_test:
    print('==> Evaluating on Test set')
    checkpoint = torch.load(model_folder_path + '/models/model_and_optim_best_checkpoint.pt')
    model.load_state_dict(checkpoint['model'])
    
    set_random_seeds(EVAL_SEED)
    
    
    test_accuracy_top1, test_total_consistency, test_string_status = evaluate_circular(model, criterion, data_loaders['test'], max_shift = MAX_SHIFT, device = device, phase = 'test')

    test_set_results = {'accuracy': test_accuracy_top1, 'consistency': test_total_consistency }

    f = open(model_folder_path+'/test_set_results.txt', 'a')
    f.write( test_string_status )
    f.close()
    pickle.dump( test_set_results, open(model_folder_path+'/test_set_results.p', "wb" ) )



                                  
if __name__ == '__main__':
    main()
             
                                                










# %%

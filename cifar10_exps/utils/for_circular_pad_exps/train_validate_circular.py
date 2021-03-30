# %%
import torch

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import time

from utils.plot_functions import plot_save_train_val_list


# %%
def train_validate_circular(model, criterion, data_loader, train_flag, optimizer, 
                device, data_augmentation_flag = False, validate_consistency = False, max_shift = 3 ):

    num_batches = len(data_loader)
    running_loss = 0.0
    running_accuracy_top1 = 0.0
    
    running_consistency = 0.0
    
    
    if train_flag == 'train':
    
        for i, data in enumerate(data_loader, 0):

            image_batch, ground_truth_labels = data
            
            image_batch = image_batch.to(device)
            ground_truth_labels = ground_truth_labels.to(device)
            
            if data_augmentation_flag == True:
                shift = np.random.randint(-max_shift, max_shift, 2)
                image_batch = torch.roll(image_batch, shifts = (shift[0], shift[1]), dims = (2,3))


            output_labels = model(image_batch)
            loss = criterion(output_labels, ground_truth_labels)

            accurate_labels = (torch.argmax(output_labels, dim = 1) ==  ground_truth_labels).tolist()
            accuracy_top1 = np.sum(accurate_labels)/len(accurate_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_accuracy_top1 += accuracy_top1
        
        epoch_loss = running_loss/num_batches
        epoch_accuracy_top1 = running_accuracy_top1/num_batches

        return epoch_loss, epoch_accuracy_top1

            

    elif train_flag == 'val':
        
        for i, data in enumerate(data_loader, 0):
            
            image_batch, ground_truth_labels = data

            image_batch = image_batch.to(device)
            ground_truth_labels = ground_truth_labels.to(device)

            output_labels = model(image_batch)
            loss = criterion(output_labels, ground_truth_labels)

            accurate_labels = (torch.argmax(output_labels, dim = 1) ==  ground_truth_labels).tolist()
            
            accuracy_top1 = np.sum(accurate_labels)/len(accurate_labels)
            
            running_loss = running_loss + loss.item()
            running_accuracy_top1 = running_accuracy_top1 + accuracy_top1
            
            
            if validate_consistency==True:
                shift = np.random.randint(-max_shift, max_shift, 2)
                
                shifted_image_batch = torch.roll(image_batch, shifts = (shift[0], shift[1]), dims = (2, 3))
                shifted_output_labels = model(shifted_image_batch)
                consistent_labels = (torch.argmax(output_labels, dim = 1) ==  torch.argmax(shifted_output_labels, dim = 1)).tolist()
                
                consistency = np.sum(consistent_labels)/len(consistent_labels)
                
                running_consistency += consistency
                

    
        epoch_loss = running_loss/num_batches
        epoch_accuracy_top1 = running_accuracy_top1/num_batches
        
        if validate_consistency == True:
            epoch_consistency = running_consistency/num_batches
            return epoch_loss, epoch_accuracy_top1, epoch_consistency
        
        else:
            return epoch_loss, epoch_accuracy_top1


def train_and_validate_circular_all_epochs(model, criterion, data_loaders, optimizer, scheduler, 
                                  scheduler_flag, start_epoch, num_epochs, 
                                  model_folder, best_accuracy, data_augmentation_flag, validate_consistency, max_shift, model_folder_path, device):

    training_loss_list = []
    validation_loss_list = []

    training_accuracy_list = []
    validation_accuracy_list = []
    best_validation_accuracy_list = []
    
    validation_consistency_list = []

    epoch_time = -1
    checkpoint = False


    for epoch in range(start_epoch, num_epochs):

        epoch_begin = time.time()

    #     train
        model.train()
        
        train_loss, train_accuracy = train_validate_circular(model, criterion, data_loaders['train'], 
                                                    train_flag = 'train', optimizer = optimizer,
                                                    data_augmentation_flag = data_augmentation_flag, max_shift = max_shift, device = device)

        training_loss_list.append(train_loss)
        training_accuracy_list.append(train_accuracy)

        if scheduler_flag == True:
            scheduler.step()

            
    #     validate
        model.eval()

        with torch.set_grad_enabled(False):

            validation_results = train_validate_circular(model, criterion, data_loaders['val'], 
                                                    train_flag = 'val', optimizer = None, validate_consistency = validate_consistency, max_shift = max_shift, device = device)
            
            validation_loss, validation_accuracy = validation_results[0:2]

            validation_loss_list.append(validation_loss)
            validation_accuracy_list.append(validation_accuracy)
            
            if validate_consistency == True:
                validation_consistency_list.append(validation_results[2])
            
            if validation_accuracy > best_accuracy:
                
                checkpoint = True
                best_accuracy = validation_accuracy
                
            best_validation_accuracy_list.append(best_accuracy)
            


    #     save everything

        plot_save_train_val_list(training_loss_list, model_folder_path +'/training_loss_list', numpy_flag = True)
        plot_save_train_val_list(training_accuracy_list, model_folder_path +'/training_accuracy_list', numpy_flag = True)
        plot_save_train_val_list(validation_loss_list, model_folder_path +'/validation_loss_list', numpy_flag = True)
        plot_save_train_val_list(validation_accuracy_list, model_folder_path +'/validation_accuracy_list', numpy_flag = True)

        plot_save_train_val_list(best_validation_accuracy_list, model_folder_path +'/best_validation_accuracy_list', numpy_flag = True)
        
        if validate_consistency == True:
            plot_save_train_val_list(validation_consistency_list, model_folder_path +'/validation_consistency_list', numpy_flag = True)
    
        state = { 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 
                 'accuracy': validation_accuracy, 'best_accuracy': best_accuracy , 'epoch': epoch }
        
        torch.save(state, model_folder_path +'/models/model_and_optim_checkpoint.pt')
        
        if checkpoint == True:
            
            state_checkpoint = { 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 
                                'accuracy': validation_accuracy, 'best_accuracy': best_accuracy, 'epoch': epoch}
            
            torch.save(state_checkpoint,model_folder_path +'/models/model_and_optim_best_checkpoint.pt')
            checkpoint = False


        epoch_time = time.time() - epoch_begin

        string = 'Epoch time: ' +str(epoch_time)+' seconds \n'
        string += str(epoch)+ ' Training loss: '+ str(training_loss_list[-1])+'\n'
        string += str(epoch)+ ' Training accuracy: '+ str(training_accuracy_list[-1])+'\n\n'
        string += str(epoch)+ ' Validation loss: '+ str(validation_loss_list[-1])+'\n'
        string += str(epoch)+ ' Validation accuracy: '+ str(validation_accuracy_list[-1])+'\n'
        string += 'Best achieved validation accuracy until now: '+str(best_validation_accuracy_list[-1])+'\n\n'
        
        if validate_consistency == True:
            string += str(epoch)+ ' Validation consistency: '+ str(validation_consistency_list[-1])+'\n\n\n'


        print(string)


        f = open(model_folder_path+'/train_val_epoch_status.txt', 'a')
        f.write( string )
        f.close()


    return training_loss_list, validation_loss_list, training_accuracy_list, validation_accuracy_list, validation_consistency_list, best_validation_accuracy_list



    

    

# %%
def evaluate_circular(model, criterion, data_loader, max_shift, device, phase):
    
    num_batches = len(data_loader)
    running_loss = 0.0
    running_accuracy_top1 = 0.0
    
    running_consistency = 0.0
    
    model.eval()
    with torch.set_grad_enabled(False):
        for i, data in enumerate(data_loader, 0):

            image_batch, ground_truth_labels = data
            image_batch = image_batch.to(device)
            ground_truth_labels = ground_truth_labels.to(device)

            random_shift1 = np.random.randint(-max_shift, max_shift, 2)
            random_shift2 = np.random.randint(-max_shift, max_shift, 2)

            shifted_image_batch1 = torch.roll(image_batch, shifts = (random_shift1[0], random_shift1[1]), dims = (2, 3))
            shifted_image_batch2 = torch.roll(image_batch, shifts = (random_shift2[0], random_shift2[1]), dims = (2, 3))

            output_labels = model(image_batch)
            shifted_output_labels1 = model(shifted_image_batch1)
            shifted_output_labels2 = model(shifted_image_batch2)

            accurate_labels = (torch.argmax(output_labels, dim = 1) ==  ground_truth_labels).tolist()
            consistent_labels = (torch.argmax(shifted_output_labels1, dim = 1) ==  torch.argmax(shifted_output_labels2, dim = 1)).tolist()

            accuracy_top1 = np.sum(accurate_labels)/len(accurate_labels)
            consistency = np.sum(consistent_labels)/len(consistent_labels)

            running_accuracy_top1 += accuracy_top1
            running_consistency += consistency


    
    total_avg_accuracy_top1 = running_accuracy_top1/num_batches
    total_avg_consistency = running_consistency/num_batches
    
    string = phase+' Accuracy: '+str(total_avg_accuracy_top1)+'\n'
    string+=phase+' Consistency: '+str(total_avg_consistency)+'\n\n'
    
    print(string)
    
    return total_avg_accuracy_top1, total_avg_consistency, string
    
    
def evaluate_circular_flip(model, criterion, data_loader, max_shift, device, phase):
#     evaluate the models on vertically and horizontally flipped cifar-10 set

    num_batches = len(data_loader)
    running_loss = 0.0
    running_accuracy_top1 = 0.0
    
    running_consistency = 0.0
    
    model.eval()
    with torch.set_grad_enabled(False):
        for i, data in enumerate(data_loader, 0):

            image_batch, ground_truth_labels = data
            image_batch = image_batch.to(device)
            
            image_batch = torch.flip(image_batch, dims = [2,3])
            
            ground_truth_labels = ground_truth_labels.to(device)

            random_shift1 = np.random.randint(-max_shift, max_shift, 2)
            random_shift2 = np.random.randint(-max_shift, max_shift, 2)

            shifted_image_batch1 = torch.roll(image_batch, shifts = (random_shift1[0], random_shift1[1]), dims = (2, 3))
            shifted_image_batch2 = torch.roll(image_batch, shifts = (random_shift2[0], random_shift2[1]), dims = (2, 3))

            output_labels = model(image_batch)
            shifted_output_labels1 = model(shifted_image_batch1)
            shifted_output_labels2 = model(shifted_image_batch2)

            accurate_labels = (torch.argmax(output_labels, dim = 1) ==  ground_truth_labels).tolist()
            consistent_labels = (torch.argmax(shifted_output_labels1, dim = 1) ==  torch.argmax(shifted_output_labels2, dim = 1)).tolist()

            accuracy_top1 = np.sum(accurate_labels)/len(accurate_labels)
            consistency = np.sum(consistent_labels)/len(consistent_labels)

            running_accuracy_top1 += accuracy_top1
            running_consistency += consistency


    
    total_avg_accuracy_top1 = running_accuracy_top1/num_batches
    total_avg_consistency = running_consistency/num_batches
    
    string = phase+' Accuracy: '+str(total_avg_accuracy_top1)+'\n'
    string+=phase+' Consistency: '+str(total_avg_consistency)+'\n\n'
    
    print(string)
    
    return total_avg_accuracy_top1, total_avg_consistency, string



def evaluate_circular_random_erase(model, criterion, data_loader, max_shift, device, random_erase_patch, phase):
    
    num_batches = len(data_loader)
    running_loss = 0.0
    running_accuracy_top1 = 0.0
    
    running_consistency = 0.0
    
    model.eval()
    with torch.set_grad_enabled(False):
        for i, data in enumerate(data_loader, 0):

            image_batch, ground_truth_labels = data
            image_batch = image_batch.to(device)
            
            erase_loc = np.random.randint(0, image_batch.shape[2] - 3 - random_erase_patch , 2)
            
            image_batch[:, :, erase_loc[0] : erase_loc[0] + random_erase_patch, erase_loc[1] : erase_loc[1] + random_erase_patch] = 0.0
            
            
            ground_truth_labels = ground_truth_labels.to(device)

            random_shift1 = np.random.randint(-max_shift, max_shift, 2)
            random_shift2 = np.random.randint(-max_shift, max_shift, 2)

            shifted_image_batch1 = torch.roll(image_batch, shifts = (random_shift1[0], random_shift1[1]), dims = (2, 3))
            shifted_image_batch2 = torch.roll(image_batch, shifts = (random_shift2[0], random_shift2[1]), dims = (2, 3))

            output_labels = model(image_batch)
            shifted_output_labels1 = model(shifted_image_batch1)
            shifted_output_labels2 = model(shifted_image_batch2)

            accurate_labels = (torch.argmax(output_labels, dim = 1) ==  ground_truth_labels).tolist()
            consistent_labels = (torch.argmax(shifted_output_labels1, dim = 1) ==  torch.argmax(shifted_output_labels2, dim = 1)).tolist()

            accuracy_top1 = np.sum(accurate_labels)/len(accurate_labels)
            consistency = np.sum(consistent_labels)/len(consistent_labels)

            running_accuracy_top1 += accuracy_top1
            running_consistency += consistency


    
    total_avg_accuracy_top1 = running_accuracy_top1/num_batches
    total_avg_consistency = running_consistency/num_batches
    
    string = phase+' Accuracy: '+str(total_avg_accuracy_top1)+'\n'
    string+=phase+' Consistency: '+str(total_avg_consistency)+'\n\n'
    
    print(string)
    
    return total_avg_accuracy_top1, total_avg_consistency, string
    
    

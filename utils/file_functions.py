import os
import sys

# def create_folders(path, model_folder):
    
#     os.chdir(path)

#     os.mkdir(model_folder)
#     os.chdir(model_folder)
    
#     os.mkdir('models')
#     os.mkdir('train_images_individual')
#     os.mkdir('validation_images_individual')
#     os.mkdir('test_images_individual')
#     os.mkdir('test_files_individual')
#     os.chdir('..')
#     os.chdir('..')




def create_folders_direct_path(model_folder_path):
    
    os.mkdir(model_folder_path)
    
    os.mkdir(model_folder_path+'/models')
    os.mkdir(model_folder_path+'/train_images_individual')
    os.mkdir(model_folder_path+'/validation_images_individual')
    os.mkdir(model_folder_path+'/test_images_individual')
    os.mkdir(model_folder_path+'/test_files_individual')



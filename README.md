# Truly shift-invariant convolutional neural networks 
### <b>Authors:</b> Anadi Chaman and Ivan Dokmanić

Despite the presence of convolutions, CNNs can be highly unstable to shifts in their input due to the presence of downsampling operations (typically in the form of strided convolutions and pooling layers). We propose **adaptive polyphase sampling (APS)**, the first approach that can restore *perfect shift invariance* in convolutional neural networks. 

This repository contains the code to implement
1. **Truly shift-invariant CNN classfiers** <a href = 'https://arxiv.org/pdf/2011.14214.pdf'>[Paper]</a> <a href = 'https://github.com/achaman2/truly_shift_invariant_cnns/files/7307076/cvpr_shift_invariant_cnns_poster.pdf'>[Poster]</a> <a href = "https://www.youtube.com/watch?v=l2jDxeaSwTs">[Video]</a>
2. **Truly shift-equivariant U-Nets for image reconstruction tasks** <a href = 'https://arxiv.org/pdf/2105.04040.pdf'>[Paper]</a> <a href = 'https://github.com/achaman2/truly_shift_invariant_cnns/files/7307089/asilomar_poster_submission.pdf'>[Poster]</a> <a href = 'https://github.com/achaman2/truly_shift_invariant_cnns/files/7309651/shift_equivariant_unet_slides.pptx'>[Slides]</a> 

### Citation
```BibTeX

@InProceedings{Chaman_2021_CVPR,
    author    = {Chaman, Anadi and Dokmanic, Ivan},
    title     = {Truly Shift-Invariant Convolutional Neural Networks},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {3773-3783}
}

 @inproceedings{Chaman_2021_equivariant,
	title={Truly shift-equivariant convolutional neural networks with adaptive polyphase upsampling},
	author={Chaman, Anadi and Dokmani{\'c}, Ivan},
	booktitle={55th Asilomar Conference on Signals, Systems, and Computers},
	year={2021}
}

```



## Loading shift invariant models
A shift invariant CNN classifier can be initialized as follows.
```python
import models.aps_models as aps_models
resnet_model = aps_models.resnet18(filter_size = 1)
```
```filter_size = j``` can be used to combine APS with anti-aliasing filters of size jxj.

Load a shift equivariant U-Net with the following commands.
```python
import models.aps_models as aps_models
unet_model = aps_models.unet_aps.unet_model.UNet_4down_aps(filter_size = 1)
```




## Training 

### Shift invariant CNN classifiers <a href = 'https://arxiv.org/pdf/2011.14214.pdf'>[Paper]</a> <a href = 'https://github.com/achaman2/truly_shift_invariant_cnns/files/7307076/cvpr_shift_invariant_cnns_poster.pdf'>[Poster]</a> <a href = "https://www.youtube.com/watch?v=l2jDxeaSwTs">[Video]</a>
We replace the downsampling operations in pooling and strided convolutions with APS layers. Thereafter, a shift in the network's input always results in a shift in its feature maps. Global average pooling layers in the end then enable perfect shift invariance. 

Below are the instructions to train models with APS using PyTorch.

**ImageNet training**

To train ResNet-18 model with APS on ImageNet use the following commands (training and evaluation with circular shifts).
```python
python3 main.py --out-dir OUT_DIR --arch resnet18_aps1 --seed 0 --data PATH-TO-DATASET
```

For training on multiple GPUs:
```python
python3 main.py --out-dir OUT_DIR --arch resnet18_aps1 --seed 0 --data PATH-TO-DATASET --workers NUM_WORKERS --dist-url tcp://127.0.0.1:FREE-PORT --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0
```
```--arch``` is used to specify the architecture. To use ResNet18 with APS layer and blur filter of size j, pass 'resnet18_apsj' as the argument to ```--arch```. List of currently supported network architectures are [here](/imagenet_exps/supported_architectures.txt). ```--circular_data_aug``` can be used to additionally train the networks with random circular shifts. Results are saved in OUT_DIR. 

**CIFAR-10 training** 

The following commands run our implementation on CIFAR-10 dataset.

```python
cd cifar10_training
python3 main.py --arch 'resnet18_aps' --filter_size j --validate_consistency --seed_num 0 --device_id 0 --model_folder CURRENT_MODEL_DIRECTORY --results_root_path ROOT_DIRECTORY --dataset_path PATH-TO-DATASET
```
```--data_augmentation_flag``` can be used to additionally train the networks with randomly shifted images. Filter size ```j``` can take the values between 1 to 7. The list of CNN architectures currently supported can be found [here](/cifar10_exps/supported_architectures.txt). The results are saved in the path: ROOT_DIRECTORY/CURRENT_MODEL_DIRECTORY/



 
### Shift equivariant U-Net for image-to-image regression tasks <a href = 'https://arxiv.org/pdf/2105.04040.pdf'>[Paper]</a> <a href = 'https://github.com/achaman2/truly_shift_invariant_cnns/files/7307089/asilomar_poster_submission.pdf'>[Poster]</a> <a href = 'https://github.com/achaman2/truly_shift_invariant_cnns/files/7309651/shift_equivariant_unet_slides.pptx'>[Slides]</a> 
To obtain shift equivariance in symmetric encoder-decoder architectures like U-Net, we propose adaptive polyphase upsampling (APS-U). With experiments on MRI and CT reconstruction tasks, we obtain state-of-the-art shift equivariance results without sacrificing on image reconstruction quality.

**Training shift equivariant U-Net on FastMRI dataset**

To train U-Net model with APS on fastMRI dataset use the following commands (training and evaluation with circular shifts).
```python
cd equivariant_unet_training/mri_reconstruction
python3 main_mri.py --arch UNet_4down_aps --out-dir OUT_DIR --data PATH-TO-FASTMRI-DATASET 
```
**Training on LoDoPaB-CT dataset**

To train a U-Net on LoDoPaB-CT dataset for CT reconstruction task
```python
cd equivariant_unet_training/ct_reconstruction
python3 main_ct.py --arch UNet_4down_aps --out-dir OUT_DIR --data PATH-TO-CT-DATASET  
```





# Truly shift-invariant convolutional neural networks 
<b>Authors:</b> Anadi Chaman and Ivan Dokmanić

Despite the presence of convolutions, popular CNN architectures are not shift invariant. For example, the performance of a CNN classifier can be impacted with a mere 1-pixel shift in its input. This is due to the presence of downsampling layers in the form of pooling/stride operations.

![unstable_downsampling](https://user-images.githubusercontent.com/12958446/136464199-d858b4b5-3d09-43a3-9a33-eb6393e409db.png)

We propose **Adaptive Polyphase Sampling (APS)**, an easy-to-implement downsampling scheme that can enable perfect shift invariance without sacrificing performance. The key idea behind APS is that sampling can be done adaptively in an input-dependent manner. We show that by choosing the sampling grid that supports pixels with the highest norm, the result can be made stable to shifts.

![aps](https://user-images.githubusercontent.com/12958446/136465122-7ff01247-52eb-453f-bbce-3b421a78bc67.png)

We use APS to restore shift invariance in CNN classifiers and shift equivariance in U-Net used for image reconstruction.

# Shift invariant CNN classifiers <a href = 'https://arxiv.org/pdf/2011.14214.pdf'>[Paper]</a> <a href = 'https://github.com/achaman2/truly_shift_invariant_cnns/files/7307076/cvpr_shift_invariant_cnns_poster.pdf'>[Poster]</a> <a href = "https://www.youtube.com/watch?v=l2jDxeaSwTs">[Video]</a>
We replace the downsampling operations in pooling and strided convolutions with APS layers. Thereafter, a shift in the network's input always results in a shift in its feature maps. Global average pooling layers in the end then enable perfect shift invariance. 

With APS, the resulting CNNs are **provably 100% shift invariant** without any loss in classification performance. In fact, the networks exhibit perfect consistency even before training, making it the first approach that makes CNNs *truly shift-invariant*. 
 
# Shift equivariant U-Net for image-to-image regression tasks <a href = 'https://arxiv.org/pdf/2105.04040.pdf'>[Paper]</a> <a href = 'https://github.com/achaman2/truly_shift_invariant_cnns/files/7307089/asilomar_poster_submission.pdf'>[Poster]</a> 
To obtain shift equivariance in symmetric encoder-decoder architectures like U-Net, we propose adaptive polyphase upsampling (APS-U). With experiments on MRI and CT reconstruction tasks, we obtain state-of-the-art shift equivariance results without sacrificing on image reconstruction quality.

# Instructions to train the networks
This repository contains our code in PyTorch to implement APS.

**ImageNet training**

To train ResNet-18 model with APS on ImageNet use the following commands (training and evaluation with circular shifts).
```
cd imagenet_exps
python3 main.py --out-dir OUT_DIR --arch resnet18_aps1 --seed 0 --data PATH-TO-DATASET
```

For training on multiple GPUs:
```
cd imagenet_exps
python3 main.py --out-dir OUT_DIR --arch resnet18_aps1 --seed 0 --data PATH-TO-DATASET --workers NUM_WORKERS --dist-url tcp://127.0.0.1:FREE-PORT --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0
```
```--arch``` is used to specify the architecture. To use ResNet18 with APS layer and blur filter of size j, pass 'resnet18_apsj' as the argument to ```--arch```. List of currently supported network architectures are [here](/imagenet_exps/supported_architectures.txt).

```--circular_data_aug``` can be used to additionally train the networks with random circular shifts. 

Results are saved in OUT_DIR. 

**CIFAR-10 training** 

The following commands run our implementation on CIFAR-10 dataset.

```
cd cifar10_exps
python3 main.py --arch 'resnet18_aps' --filter_size FILTER_SIZE --validate_consistency --seed_num 0 --device_id 0 --model_folder CURRENT_MODEL_DIRECTORY --results_root_path ROOT_DIRECTORY --dataset_path PATH-TO-DATASET
```
```--data_augmentation_flag``` can be used to additionally train the networks with randomly shifted images. FILTER_SIZE can take the values between 1 to 7. The list of CNN architectures currently supported can be found [here](/cifar10_exps/supported_architectures.txt).

The results are saved in the path: ROOT_DIRECTORY/CURRENT_MODEL_DIRECTORY/


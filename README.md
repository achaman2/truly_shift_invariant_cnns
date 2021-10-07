# Truly shift-invariant convolutional neural networks <a href = 'https://arxiv.org/pdf/2011.14214.pdf'>[Paper]</a> <a href = 'https://www.icloud.com/iclouddrive/0gyJb-RxZ6tCRGe91Ig09E0RA#cvpr_shift_invariant_cnns_poster'>[Poster]</a> <a href = "https://www.youtube.com/watch?v=l2jDxeaSwTs">[Video]</a>
<b>Authors:</b> Anadi Chaman and Ivan Dokmanić

Despite the presence of convolutions, popular CNN architectures are not shift invariant. For example, the performance of a CNN classifier can be impacted with a mere 1-pixel shift in its input. This is due to the presence of downsampling layers in the form of pooling/stride operations.

![unstable_downsampling](https://user-images.githubusercontent.com/12958446/136464199-d858b4b5-3d09-43a3-9a33-eb6393e409db.png)

In this work, we present <b>Adaptive Polyphase Sampling</b> (APS), an easy-to-implement non-linear downsampling scheme that completely gets rid of this problem. The resulting CNNs yield <b>100% consistency</b> in classification performance under shifts without any loss in accuracy. In fact, unlike prior works, the  networks exhibit perfect consistency even before training, making it the first approach that makes CNNs <i>truly shift invariant</i>.

This repository contains our code in PyTorch to implement APS.

# ImageNet training
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

# CIFAR-10 training
The following commands run our implementation on CIFAR-10 dataset.

```
cd cifar10_exps
python3 main.py --arch 'resnet18_aps' --filter_size FILTER_SIZE --validate_consistency --seed_num 0 --device_id 0 --model_folder CURRENT_MODEL_DIRECTORY --results_root_path ROOT_DIRECTORY --dataset_path PATH-TO-DATASET
```
```--data_augmentation_flag``` can be used to additionally train the networks with randomly shifted images. FILTER_SIZE can take the values between 1 to 7. The list of CNN architectures currently supported can be found [here](/cifar10_exps/supported_architectures.txt).

The results are saved in the path: ROOT_DIRECTORY/CURRENT_MODEL_DIRECTORY/


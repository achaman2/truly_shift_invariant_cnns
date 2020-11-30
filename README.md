# Truly shift-invariant convolutional neural networks <a href = ''>[Paper]</a> 
<b>Authors:</b> Anadi Chaman and Ivan Dokmanić

Convolutional neural networks were always assumed to be shift invariant, until recently when it was shown that the classification accuracy of a trained CNN can take a serious hit with merely a 1-pixel shift in input image. One of the primary reasons for this problem is the use of downsampling (popularly known as stride) layers in the networks.

In this work, we present <b>Adaptive Polyphase Sampling</b> (APS), an easy-to-implement non-linear downsampling scheme that completely gets rid of this problem. The resulting CNNs yield <b>100% consistency</b> in classification performance under shifts without any loss in accuracy. In fact, unlike prior works, the  networks exhibit perfect consistency even before training, making it the first approach that makes CNNs <i>truly shift invariant</i>.

This repository contains our code in PyTorch to implement APS.

# Usage
To run our implementation use the following command

```
python3 main.py --arch 'resnet18_aps' --filter_size FILTER_SIZE --validate_consistency --seed_num 0 --device_id 0 --model_folder CURRENT_MODEL_DIRECTORY --results_root_path ROOT_DIRECTORY  
```
```--data_augmentation_flag``` can be used to additionally train the networks with randomly shifted images. FILTER_SIZE can take the values between 1 to 7. The following CNN architectures are currently supported:\

<b>Vanila ResNet models:</b>  'resnet20', 'resnet56', 'resnet18', 'resnet50'\
<b> LPF ResNet models: </b>  'resnet20_lpf', 'resnet56_lpf', 'resnet18_lpf', 'resnet50_lpf' \
<b> APS ResNet models: </b>  'resnet20_aps', 'resnet56_aps', 'resnet18_aps', 'resnet50_aps' 


The results are saved in the path: ROOT_DIRECTORY/CURRENT_MODEL_DIRECTORY/


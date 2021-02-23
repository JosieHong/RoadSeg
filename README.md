<!--
 * @Author: JosieHong
 * @Date: 2021-01-31 13:25:12
 * @LastEditAuthor: JosieHong
 * @LastEditTime: 2021-02-02 15:14:40
-->
# RoadSeg_Pytorch

This is an road segmentation network of Pytorch, which is inspired by [KittiSeg](https://github.com/MarvinTeichmann/KittiSeg). 

The detailed network structure is shown in the following figure. The encoder is the ResNet50/101 provided by [Torchvision](https://pytorch.org/docs/stable/torchvision/models.html), so the details are not marked here. 

<div align="center">
	<img src="./img/network_structure.png" alt="network_structure" width="743.2">
</div>

## Set Up

1. Set up the environment by Virtualenv: 

```bash
virtualenv env
source enb/bin/activate

# install pytorch for Linux, CUDA=10.1
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
# more details could be found: https://pytorch.org/get-started/locally/

# If torch<1.5.0, you may need to install for logical calculation on GPU

pip install tqdm opencv-python
```

2. Download KITTI road dataset: http://www.cvlibs.net/download.php?file=data_road.zip

The dataset structure is shown below. Because it does not provide the ground-truth of testing data, we only use the training data. When loading the dataset, we split training data into three subsets: a) training (173 images), b) validation (58 images), and c) testing (58 images). 

```bash
|_data
    |_data_road
        |_training
        |   |_calib
        |   |_image_2
        |   |_gt_image_2
        |_testing
            |_calib
            |_image_2
```

## Train

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --dataset ./data/data_road/ --batchSize 12 --nepoch 24 --model ./checkpoints/model_23.pth
```

## Performance

Epoch 24 glob acc : 0.914, pre : 0.924, recall : 0.373, F_score : 0.531, IoU : 0.361, mIoU:  0.688

Epoch 48 glob acc : 0.926, pre : 0.923, recall : 0.506, F_score : 0.654, IoU : 0.485, mIoU:  0.777

Epoch 72 glob acc : 0.927, pre : 0.918, recall : 0.495, F_score : 0.643, IoU : 0.474, mIoU:  0.787

Epoch 96 glob acc : 0.925, pre : 0.921, recall : 0.490, F_score : 0.640, IoU : 0.470, mIoU:  0.792

## TSD-max Dataset

```bash
python train.py --dataset_type tsd --dataset ./data/tsd-max-traffic/ --batchSize 4 --nepoch 24 --imgSize (256,256)


```
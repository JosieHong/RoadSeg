<!--
 * @Author: JosieHong
 * @Date: 2021-01-31 13:25:12
 * @LastEditAuthor: JosieHong
 * @LastEditTime: 2021-02-02 15:14:40
-->
# RoadSeg

This is a road segmentation network of Pytorch, which is inspired by [KittiSeg](https://github.com/MarvinTeichmann/KittiSeg). It has been tested on the KITTI dataset (public) and the TSD-max dataset (custom). 

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

3. TSD-max Dataset:

TSD-max Dataset is constructed by the Institute of Artificial Intelligence and Robotics, Xi’an Jiaotong University. There are 979 training images and 244 test images are labeled manually. 

```
|_data
    |_train_lst.txt
    |_test_lst.txt
    |_sequence_1 (rgb images and labels)
    |_....
```

## Train

```bash
# KITTI
python train.py --dataset ./data/data_road/ --batchSize 1 --nepoch 24 --imgSize 512 512 --outf ./checkpoints/kitti
# (A tricky way to test)
python train.py --dataset ./data/data_road/ --batchSize 4 --nepoch 0 --outf ./checkpoints/kitti --model ./checkpoints/kitti/save_weighted.pth

# TSD-max
python train.py --dataset_type tsd --dataset ./data/tsd-max-traffic/ --batchSize 2 --nepoch 24 --imgSize 256 256 --outf ./checkpoints/tsd_256 --model ./checkpoints/tsd_256/model_23.pth
python train.py --dataset_type tsd --dataset ./data/tsd-max-traffic/ --batchSize 1 --nepoch 24 --imgSize 512 512 --outf ./checkpoints/tsd_512
# (A tricky way to test)
python train.py --dataset_type tsd --dataset ./data/tsd-max-traffic/ --batchSize 2 --nepoch 0 --imgSize 256 256 --outf ./checkpoints/tsd_256 --model ./checkpoints/tsd_256/save_weighted.pth
python train.py --dataset_type tsd --dataset ./data/tsd-max-traffic/ --batchSize 1 --nepoch 0 --imgSize 512 512 --outf ./checkpoints/tsd_512 --model ./checkpoints/tsd_512/save_weighted.pth
```

## Performance

| dataset | input image size | training epoch | mIoU  |
| ------- | ---------------- | -------------- | ----- |
| KITTI   | 256 x 256        | 48             | 0.752 |
| KITTI   | 512 x 512        | 24             | 0.827 |
| TSD-max | 256 x 256        | 48             | 0.870 |
| TSD-max | 512 x 512        | 24             | 0.973 |
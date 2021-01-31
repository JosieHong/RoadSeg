<!--
 * @Author: JosieHong
 * @Date: 2021-01-30 15:54:39
 * @LastEditAuthor: JosieHong
 * @LastEditTime: 2021-01-30 23:24:56
-->
# KITTI_Seg pytorch

## Set Up

1. Set up the environment by Virtualenv: 

```bash
virtualenv env
source enb/bin/activate

# install pytorch for Linux, CUDA=10.1
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
# more details could be found: https://pytorch.org/get-started/locally/
```

2. Download KITTI road dataset: http://www.cvlibs.net/download.php?file=data_road.zip

```bash
|_data
    |_data_road
        |_train
        |   |_calib
        |   |_image_2
        |   |_gt_image_2
        |_testing
            |_calib
            |_image_2
```

## Train

```bash
python train.py --dataset ./data/data_road/ --batchSize 1
```
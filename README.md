# Road_Seg

[Neurocomputing] This is an road segmentation network of Pytorch, which is inspired by [KittiSeg](https://github.com/MarvinTeichmann/KittiSeg), which is the part of the experiments of our paper, [Geometric and semantic analysis of road image sequences for traffic scene construction](https://www.sciencedirect.com/science/article/pii/S0925231221013564). 

The detailed network structure is shown in the following figure. The encoder is the ResNet50/101 provided by [Torchvision](https://pytorch.org/docs/stable/torchvision/models.html), so the details are not marked here. 

<div align="center">
	<img src="./img/network_structure.png" alt="network_structure" width="743.2">
</div>

## Performance

**1. binary class segmentation**

|       | Bg Acc     | Road Acc   | Bg IoU     | Road IoU   |
| ----- | ---------- | ---------- | ---------- | ---------- |
| U-Net | 98.114     | 97.941     | 96.368     | 96.424     |
| GCN   | 98.654     | **98.921** | **97.661** | **97.842** |
| Ours  | **98.663** | 98.657     | 97.402     | 97.531     |

**2. semantic segmentation**

|       | Road IoU | Road Acc | Car IoU | Car Acc | Others IoU | Others Acc |
| ----- | -------- | -------- | ------- | ------- | ---------- | ---------- |
| U-Net | 96.478   | 98.288   | 0       | 0       | 0          | 0          |
| GCN   | 97.606   | 98.926   | 23.156  | 26.970  | 0.398      | 0.413      |
| Ours  | 96.869   | 98.267   | 00.022  | 00.024  | 0          | 0          |

## Set up

```
virtuelenv env
source env/bin/activate
pip install -r requirements.txt
```

## U-Net

Reference: https://github.com/milesial/Pytorch-UNet

```bash
# train&test
python run.py --model unet --dataset TSDDataset_bin --epochs 5
# test
python run.py --model unet --dataset TSDDataset_bin --epochs 0 --load checkpoints/unet/MODEL_bin.pth
# visualization
python predict.py --model unet --dataset TSDDataset_bin --load checkpoints/unet/MODEL_bin.pth

# train&test
python run.py --model unet --dataset TSDDataset_mul --epochs 5
# test
python run.py --model unet --dataset TSDDataset_mul --epochs 0 --load checkpoints/unet/MODEL_mul.pth
# visualization
python predict.py --model unet --dataset TSDDataset_mul --load checkpoints/unet/MODEL_mul.pth
```

## GCN

Reference: https://github.com/SConsul/Global_Convolutional_Network

```bash
# train&test
python run.py --model gcn --dataset TSDDataset_bin --epochs 5
# test
python run.py --model gcn --dataset TSDDataset_bin --epochs 0 --load checkpoints/gcn/MODEL_bin.pth
# visualization
python predict.py --model gcn --dataset TSDDataset_bin --load checkpoints/gcn/MODEL_bin.pth

# train&test
python run.py --model gcn --dataset TSDDataset_mul --epochs 5
# test
python run.py --model gcn --dataset TSDDataset_mul --epochs 0 --load checkpoints/gcn/MODEL_mul.pth
# visualization
python predict.py --model gcn --dataset TSDDataset_mul --load checkpoints/gcn/MODEL_mul.pth
```

## Ours

```bash
# train&test
python run.py --model ours --dataset TSDDataset_bin --epochs 5
# test
python run.py --model ours --dataset TSDDataset_bin --epochs 0 --load checkpoints/ours/MODEL_bin.pth
# visualization
python predict.py --model ours --dataset TSDDataset_bin --load checkpoints/ours/MODEL_bin.pth

# train&test
python run.py --model ours --dataset TSDDataset_mul --epochs 3
# test
python run.py --model ours --dataset TSDDataset_mul --epochs 0 --load checkpoints/ours/MODEL_mul.pth
# visualization
python predict.py --model ours --dataset TSDDataset_mul --load checkpoints/ours/MODEL_mul.pth
```
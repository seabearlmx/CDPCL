## CDPCL (ACM MM 2023): Official Project Webpage
This repository provides the official PyTorch implementation of the following paper:
> **CDPCL:** Calibration-based Dual Prototypical Contrastive Learning Approach for Domain Generalization Semantic Segmentation<br>
> Muxin Liao*, Shishun Tian*, Yuhang Zhang, Guoguang Hua, Wenbin Zou, Xia Li (*: equal contribution)<br>
> ACM MM 2023, Accepted as Poster<br>

> Paper: [ACM Digital Library](https://dl.acm.org/doi/abs/10.1145/3581783.3611792)<br>

> **Abstract:** 
*Prototypical contrastive learning (PCL) has been widely used to learn class-wise domain-invariant features recently. These methods are based on the assumption that the prototypes, which are represented as the central value of the same class in a certain domain, are domain-invariant. Since the prototypes of different domains have discrepancies as well, the class-wise domain-invariant features learned from the source domain by PCL need to be aligned with the prototypes of other domains simultaneously. However, the prototypes of the same class in different domains may be different while the prototypes of different classes may be similar, which may affect the learning of class-wise domain-invariant features. Based on these observations, a calibration-based dual prototypical contrastive learning (CDPCL) approach is proposed to reduce the domain discrepancy between the learned class-wise features and the prototypes of different domains for domain generalization semantic segmentation. It contains an uncertainty-guided PCL (UPCL) and a hard-weighted PCL (HPCL). Since the domain discrepancies of the prototypes of different classes may be different, we propose an uncertainty probability matrix to represent the domain discrepancies of the prototypes of all the classes. The UPCL estimates the uncertainty probability matrix to calibrate the weights of the prototypes during the PCL. Moreover, considering that the prototypes of different classes may be similar in some circumstances, which means these prototypes are hard-aligned, the HPCL is proposed to generate a hard-weighted matrix to calibrate the weights of the hard-aligned prototypes during the PCL. Extensive experiments demonstrate that our approach achieves superior performance over current approaches on domain generalization segmentation tasks. The source code will be released at https://github.com/seabearlmx/CDPCL.*<br>

## Code Contributors
[Muxin Liao](https://scholar.google.com/citations?user=RVt9XHEAAAAJ&hl=zh-CN)

## Pytorch Implementation
### Installation
Clone this repository.
```
git clone https://github.com/seabearlmx/CDPCL.git
cd CDPCL
```
Install following packages.
```
conda create --name cdpcl python=3.7
conda activate cdpcl
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install scipy==1.1.0
conda install tqdm==4.46.0
conda install scikit-image==0.16.2
pip install tensorboardX
pip install thop
pip install kmeans1d
imageio_download_bin freeimage
```
### How to Run CDPCL

The running scripts and experimental settings highly rely on [RobustNet](https://github.com/shachoi/RobustNet).

We evaludated CDPCL on [Cityscapes](https://www.cityscapes-dataset.com/), [BDD-100K](https://bair.berkeley.edu/blog/2018/05/30/bdd/),[Synthia](https://synthia-dataset.net/downloads/) ([SYNTHIA-RAND-CITYSCAPES](http://synthia-dataset.net/download/808/)), [GTAV](https://download.visinf.tu-darmstadt.de/data/from_games/) and [Mapillary Vistas](https://www.mapillary.com/dataset/vistas?pKey=2ix3yvnjy9fwqdzwum3t9g&lat=20&lng=0&z=1.5).

We adopt Class uniform sampling proposed in [this paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Improving_Semantic_Segmentation_via_Video_Propagation_and_Label_Relaxation_CVPR_2019_paper.pdf) to handle class imbalance problems. [GTAVUniform](https://github.com/shachoi/RobustNet/blob/0538c69954c030273b3df952f90347572ecac53b/datasets/gtav.py#L306) and [CityscapesUniform](https://github.com/shachoi/RobustNet/blob/0538c69954c030273b3df952f90347572ecac53b/datasets/cityscapes.py#L324) are the datasets to which Class Uniform Sampling is applied.


1. For Cityscapes dataset, download "leftImg8bit_trainvaltest.zip" and "gtFine_trainvaltest.zip" from https://www.cityscapes-dataset.com/downloads/<br>
Unzip the files and make the directory structures as follows.
```
cityscapes
 └ leftImg8bit_trainvaltest
   └ leftImg8bit
     └ train
     └ val
     └ test
 └ gtFine_trainvaltest
   └ gtFine
     └ train
     └ val
     └ test
```
```
bdd-100k
 └ images
   └ train
   └ val
   └ test
 └ labels
   └ train
   └ val
```
```
mapillary
 └ training
   └ images
   └ labels
 └ validation
   └ images
   └ labels
 └ test
   └ images
   └ labels
```

#### We used [GTAV_Split](https://download.visinf.tu-darmstadt.de/data/from_games/code/read_mapping.zip) to split GTAV dataset into training/validation/test set. Please refer the txt files in [split_data](https://github.com/seabearlmx/CDPCL/tree/main/split_data).

```
GTAV
 └ images
   └ train
     └ folder
   └ valid
     └ folder
   └ test   
     └ folder
 └ labels
   └ train
     └ folder
   └ valid
     └ folder
   └ test   
     └ folder
```

#### We randomly splitted [Synthia dataset](http://synthia-dataset.net/download/808/) into train/val set. Please refer the txt files in [split_data](https://github.com/seabearlmx/CDPCL/tree/main/split_data).

```
synthia
 └ RGB
   └ train
   └ val
 └ GT
   └ COLOR
     └ train
     └ val
   └ LABELS
     └ train
     └ val
```

2. You should modify the path in **"<path_to_CDPCL>/config.py"** according to your dataset path.
```
#Cityscapes Dir Location
__C.DATASET.CITYSCAPES_DIR = <YOUR_CITYSCAPES_PATH>
#Mapillary Dataset Dir Location
__C.DATASET.MAPILLARY_DIR = <YOUR_MAPILLARY_PATH>
#GTAV Dataset Dir Location
__C.DATASET.GTAV_DIR = <YOUR_GTAV_PATH>
#BDD-100K Dataset Dir Location
__C.DATASET.BDD_DIR = <YOUR_BDD_PATH>
#Synthia Dataset Dir Location
__C.DATASET.SYNTHIA_DIR = <YOUR_SYNTHIA_PATH>
```
3. You can train CDPCL with following commands.
```
<path_to_CDPCL>$ CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/train_r50os16_gtav_isw.sh # Train: GTAV, Test: BDD100K, Cityscapes, Synthia, Mapillary / ResNet50
```

## Acknowledgments
Our pytorch implementation is heavily derived from [RobustNet](https://github.com/shachoi/RobustNet).
Thanks to the NVIDIA implementations.

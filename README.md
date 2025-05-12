# Drone-Guard: A Self-Supervised Deep Learning Framework for Spatiotemporal Anomaly Detection in UAV Surveillance Videos
This is the code for **[Drone-Guard: A Self-Supervised Deep Learning Framework for Spatiotemporal Anomaly Detection in UAV Surveillance Videos]()** [![Hits](https://github.com/slitiWassim/Drone-Guard)](https://hits.seeyoufarm.com).

### [Project](https://slitiwassim.github.io/Drone-Guard/) | [Video](https://youtu.be/s5wmFbDnBCI) | [Paper](https://link.springer.com/article/10.1007/s10489-024-06042-4?utm_source=rct_congratemailt&utm_medium=email&utm_campaign=oa_20250103&utm_content=10.1007%2Fs10489-024-06042-4)
 
## Related Works
> **CrossAnomaly**: See [CrossAnomaly: A Contextual Cross-Modality Framework for Video Anomaly Detection](https://github.com/vt-le/CrossAnomaly).

> **HSTforU**: See [HSTforU: Anomaly Detection in Aerial and Ground-based Videos with Hierarchical Spatio-Temporal Transformer for U-net](https://github.com/vt-le/HSTforU/tree/main).

> **ASTNet**: See [Attention-based Residual Autoencoder for Video Anomaly Detection](https://vt-le.github.io/astnet/).

<a href="static/videos/Bike_video.gif" target="_blank">
    <image style="border: 2px solid rgb(201, 196, 196);" src="static/videos/Bike_video.gif"  width="100%">
</a>

## Model

<a href="static/images/Model-Architecture.png" target="_blank">
    <image style="border: 2px solid rgb(201, 196, 196);" src="static/images/Model-Architecture.png" width="100%">
</a>



## Setup
The code can be run under any environment with Python 3.12 and above.
(It may run with lower versions, but we have not tested it).

Install the required packages:

    pip install -r requirements.txt
  
Clone this repo:

    git clone https://github.com/slitiWassim/Drone-Guard.git
    cd Drone-Guard/

We evaluate `Drone-Guard` on:
| Dataset | Link                                                                                  |
|--|---------------------------------------------------------------------------------------|
| UCSD Ped2 | [![Google drive](https://badgen.net/static/Homepage/Ped2/blue)](http://www.svcl.ucsd.edu/projects/anomaly/dataset.html) |
| CUHK Avenue | [![Google drive](https://badgen.net/badge/Homepage/Avenue/cyan)](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html) |
| ShanghaiTech | [![Google drive](https://badgen.net/badge/Homepage/ShanghaiTech/green?)](https://svip-lab.github.io/dataset/campus_dataset.html) |
| Drone-Anomaly | [![Google drive](https://badgen.net/badge/Homepage/Drone-Anomaly/yellow)](https://github.com/Jin-Pu/Drone-Anomaly)    |

A dataset is a directory with the following structure:
  ```bash
  $ tree data
  ped2/avenue
  ├── training
  │   └── frames
  │       ├── ${video_1}$
  │       │   ├── 000.jpg
  │       │   ├── 001.jpg
  │       │   └── ...
  │       ├── ${video_2}$
  │       │   ├── 00.jpg
  │       │   └── ...
  │       └── ...
  └── testing
      └── frames
          ├── ${video_1}$
          │   ├── 000.jpg
          │   ├── 001.jpg
          │   └── ...
          ├── ${video_2}$
          │   ├── 000.jpg
          │   └── ...
          └── ...
 
  
  shanghaitech
  ├── training
  │   └── frames
  │       ├── ${video_1}$
  │       │   ├── 000.jpg
  │       │   ├── 001.jpg
  │       │   └── ...
  │       ├── ${video_2}$
  │       │   ├── 00.jpg
  │       │   └── ...
  │       └── ...
  └── testing
      └── frames
          ├── ${video_1}$
          │   ├── 000.jpg
          │   ├── 001.jpg
          │   └── ...
          ├── ${video_2}$
          │   ├── 000.jpg
          │   └── ...
          └── ...


  drone
  ├──bike
  │  ├──training
  │  │  └── frames
  │  │      ├── ${video_1}$
  │  │      │   ├── 0.jpg
  │  │      │   ├── 1.jpg
  │  │      │   └── ...
  │  │      ├── ${video_2}$
  │  │      │   ├── 00.jpg
  │  │      │   └── ...
  │  │      └── ...
  │  ├── testing
  │  │   └── frames
  │  │       ├── ${video_1}$
  │  │       │   ├── 000.jpg
  │  │       │   ├── 001.jpg
  │  │       │   └── ...
  │  │       ├── ${video_2}$
  │  │       │   ├── 000.jpg
  │  │       │   └── ...
  │  │       └── ...
  │  └── annotation
  │      ├── 01.npy
  │      ├── 02.npy
  │      └── ...
  ├── highway
  │   ├── ...
  └── ...
  
  ```

## Training
To train `Drone-Guard` on a dataset, run:
```bash
 python  train.py --cfg <config-file> --pseudo True
```  
 For example, to train `Drone-Guard` on Ped2:

```bash
python train.py \
    --cfg config/ped2.yaml \
    --pseudo True # To Train model with both normal and pseudo anomalies data
```


## Evaluation
Please first download the pre-trained model

| Dataset | Pretrained Model                                                                                  |
|--|---------------------------------------------------------------------------------------|
| UCSD Ped2 | [![Google drive](https://badgen.net/static/Link/Ped2/blue?icon=chrome)](https://drive.google.com/file/d/1M2zmfCxYB-f7e9zxoDzVeYxA7bcF8ock/view?usp=drive_link) |
| CUHK Avenue | [![Google drive](https://badgen.net/badge/Link/Avenue/blue?icon=chrome)](https://drive.google.com/file/d/1UkPwkGvONeGrv12lyJ2f0VCUcfiCbxiv/view?usp=drive_link) |
| ShanghaiTech | [![Google drive](https://badgen.net/badge/Link/ShanghaiTech/blue?icon=chrome)](https://drive.google.com/file/d/1bxOlFPju_LONHjJQlUo28dIpY8ZSucto/view?usp=drive_link) |
| Drone-Anomaly | [![Google drive](https://badgen.net/badge/Link/Drone-Anomaly/blue?icon=chrome)](https://drive.google.com/drive/folders/1MyzMWROIyj7iAHPFZmg1RD7QmW0CHzTU?usp=drive_link)    |

To evaluate a pretrained `Drone-Guard` on a dataset, run:

```bash
 python test.py \
    --cfg <path/to/config/file> \
    --pretrained </path/to/pre-trained/model> 
```      
 
 For example, to evaluate `Drone-Guard` on Ped2:

```bash
python test.py \
    --cfg config/ped2.yaml \
    --model-file  pre-trained/best_model_ped2.pth
```
<!-- 
## Training from scratch
To train `HSTforU` on a dataset, run:
```bash
python -m torch.distributed.launch \
    --nproc_per_node <num-of-gpus-to-use> \
    --master_port 12345  main.py \ 
    --cfg <path/to/config/file> \
    [--batch-size <batch-size-per-gpu> --tag <job-tag>]
```

For example, to train `HSTforU` on Ped2:

```bash
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 train.py --cfg configs/scripts/ped2/ped2_pvt2_hst.yaml 
``` 
-->
## Configuration
 * We use [YAML](https://yaml.org/) for configuration.
 * We provide a couple preset configurations.
 * Please refer to `config.py` for documentation on what each configuration does.

## Citing
If you find our work useful, please consider citing:
```BibTeX
﻿@Article{}
```

## Contact
For any question, please file an [issue](https://github.com/slitiWassim/Drone-Guard/issues) or contact:

    Wassim Sliti : wassim.sliti@ept.ucar.tn

## Acknowledgement

The code is built on top of code provided by Le et al. [ github ](https://github.com/vt-le/astnet.git)  
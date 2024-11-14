

# DenseMAE: Encoder-driven dense decoding makes autoencoders stronger for medical image segmentation



## TODOs

- [x] Code released
- [ ] Weight released



## Getting Started

### Prepare Environment

```
conda create -n DenseMAE python=3.9
conda activate DenseMAE
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0
pip install packaging timm==0.5.4
pip install transformers==4.34.1 typed-argument-parser
pip install numpy==1.21.2 opencv-python==4.5.5.64 opencv-python-headless==4.5.5.64
pip install 'monai[all]'
pip install monai==1.2.0
```

### Prepare Datasets

We recommend you to convert the dataset into the nnUNet format.

```
└── DenseMAE
    ├── data
        ├── Dataset001_BTCV
            └── imagesTr
                ├── xxx_0000.nii.gz
                ├── ...
        ├── Dataset006_FLARE2022
            └── imagesTr
                ├── xxx_0000.nii.gz
                ├── ...
        └── Other_dataset
            └── imagesTr
                ├── xxx_0000.nii.gz
                ├── ...
```



## Start Training

Run training on multi-GPU :

```sh
# An example of training on 4 GPUs with DDP
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12351 main.py
```



## Fine-tuning

Load pre-training weights :

```python
# An example of Fine-tuning on BTCV (num_classes=14)
from downstream.factory import load_dense_mae_10k

model = load_dense_mae_10k(n_classes=14)
```

# vq-vae-2-pytorch
Implementation of Generating Diverse High-Fidelity Images with VQ-VAE-2 in PyTorch

## Update

* 2020-06-01

train_vqvae.py and vqvae.py now supports distributed training. You can use --n_gpu [NUM_GPUS] arguments for train_vqvae.py to use [NUM_GPUS] during training.

## Requisite

* Python >= 3.6
* PyTorch >= 1.1
* lmdb (for storing extracted codes)

[Checkpoint of VQ-VAE pretrained on FFHQ](vqvae_560.pt)

## Usage

Currently supports 256px (top/bottom hierarchical prior)

1. Stage 1 (VQ-VAE)

> python train_vqvae.py [DATASET PATH]

If you use FFHQ, I highly recommends to preprocess images. (resize and convert to jpeg)

2. Extract codes for stage 2 training

> python extract_code.py --ckpt checkpoint/[VQ-VAE CHECKPOINT] --name [LMDB NAME] [DATASET PATH]

3. Stage 2 (PixelSNAIL)

> python train_pixelsnail.py [LMDB NAME]

Maybe it is better to use larger PixelSNAIL model. Currently model size is reduced due to GPU constraints.

## Sample

### Stage 1

Note: This is a training sample

![Sample from Stage 1 (VQ-VAE)](stage1_sample.png)

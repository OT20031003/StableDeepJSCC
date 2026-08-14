# ADJSCC / BDJSCC — PyTorch implementation

This directory contains a TensorFlow-free PyTorch implementation of
**Wireless Image Transmission Using Deep Source Channel Coding with Attention
Modules**. Both the adaptive model (ADJSCC) and baseline model (BDJSCC) are
available for CIFAR-10 and ImageNet/Kodak.

## What is implemented

- Analysis/synthesis transforms with GDN and IGDN
- SNR-aware attention-feature (AF) modules
- Optional fading-coefficient-aware AF modules
- Per-codeword power normalization
- AWGN, slow fading, equalized slow fading, and Bernoulli-Gaussian burst noise
- CIFAR-10 training and evaluation
- ImageNet training from an image directory or the legacy
  `Imagenet_patch_128.record` file
- Kodak SNR sweeps and single-image reconstruction
- Native PyTorch `.pt` checkpoints and resumable Adam optimizer state

All image tensors use NCHW layout and values in `[0, 1]`. Reported MSE is in
that normalized range; PSNR uses a data range of 1, so the resulting PSNR is
equivalent to computing MSE on `[0, 255]` images with a data range of 255.

## Environment

```bash
conda activate ldm
```

ADJSCC uses the repository's existing [`environment.yaml`](../environment.yaml)
without replacing or upgrading any of its packages. It targets the versions
pinned there:

- Python 3.8.5
- PyTorch 1.11.0
- torchvision 0.12.0
- NumPy 1.19.2
- CUDA Toolkit 11.3 (CPU execution is also supported)

No additional package installation is required. In particular, do not install
a separate PyTorch, torchvision, NumPy, TensorFlow, TensorFlow Probability,
TensorFlow Compression, CompressAI, or protobuf package for ADJSCC. Pillow is
already supplied as a torchvision dependency in the `ldm` environment.

## Datasets

### CIFAR-10

The CIFAR-10 scripts download the dataset through torchvision by default. Use
`--no-download` when the data is already present or network access is disabled.

### ImageNet

Pass either of the following to `--data-dir`:

1. A directory containing JPEG/PNG images at any nesting depth.
2. The legacy `Imagenet_patch_128.record` TFRecord file.
3. A directory containing `Imagenet_patch_128.record`.

The included lightweight TFRecord reader understands the original
`tf.train.Example` records and extracts `image/encoded` without importing
TensorFlow.

### Kodak

Place the 24 images at:

```text
ADJSCC/dataset/kodak/kodim01.png
...
ADJSCC/dataset/kodak/kodim24.png
```

or pass another directory with `--kodak-dir`.

## Usage

Activate `ldm`, then run commands from the repository root. Use `--device auto`
(the default), `--device cuda`, or `--device cpu`.

### Adaptive model on CIFAR-10

```bash
python ADJSCC/adjscc_cifar10.py train --channel-type awgn
python ADJSCC/adjscc_cifar10.py eval --channel-type awgn
python ADJSCC/adjscc_cifar10.py eval_burst \
  --channel-type awgn --burst-snr-eval 10 --burst-stddev 1.0
```

### Baseline model on CIFAR-10

```bash
python ADJSCC/bdjscc_cifar10.py train --snr-train 10
python ADJSCC/bdjscc_cifar10.py train_mix --snr-low-mix 0 --snr-up-mix 20
python ADJSCC/bdjscc_cifar10.py eval_mismatch --snr-train 10
```

### Adaptive model on ImageNet/Kodak

```bash
python ADJSCC/adjscc_imagenet.py train --data-dir /path/to/imagenet
python ADJSCC/adjscc_imagenet.py eval --kodak-dir /path/to/kodak
python ADJSCC/adjscc_imagenet.py predict \
  --predict-image /path/to/image.png --snr-predict 10
```

### Baseline model on ImageNet/Kodak

```bash
python ADJSCC/bdjscc_imagenet.py train --data-dir /path/to/imagenet
python ADJSCC/bdjscc_imagenet.py train_mix --data-dir /path/to/imagenet
python ADJSCC/bdjscc_imagenet.py eval_mismatch --kodak-dir /path/to/kodak
python ADJSCC/bdjscc_imagenet.py eval_pic --eval-image /path/to/image.png
```

Use `python ADJSCC/<script>.py --help` for every option. For a quick smoke run,
`--feature-channels`, `--max-train-batches`, and `--max-eval-batches` can reduce
the model size and processed batches; the paper-compatible feature width is the
default value of 256.

## Checkpoint compatibility

The old Keras `.h5` files are not directly loadable by PyTorch. This repository
does not contain pretrained weights, so the PyTorch models should normally be
trained from scratch. New checkpoints use `.pt` and include model weights,
optimizer state, epoch, best loss, and run metadata.

## Citation

J. Xu, B. Ai, W. Chen, A. Yang, P. Sun and M. Rodrigues, “Wireless Image
Transmission Using Deep Source Channel Coding With Attention Modules,” *IEEE
Transactions on Circuits and Systems for Video Technology*, vol. 32, no. 4,
pp. 2315–2328, April 2022, doi: 10.1109/TCSVT.2021.3082521.

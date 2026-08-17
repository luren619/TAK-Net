# TAK-Net

Official PyTorch implementation of **TAK-Net**, a paired B-mode/contrast-enhanced
ultrasound (CEUS) network for carotid wall segmentation in Takayasu arteritis.

TAK-Net contains three task-oriented components:

- **DAF** adaptively fuses aligned B-mode and CEUS features at corresponding encoder levels.
- **MCE** aggregates full-scale features and calibrates multi-scale spatial context.
- **LGD** predicts a detached localization map to guide decoding and adds training-time auxiliary supervision.

## Repository scope

This repository contains only the code needed to train, test, and evaluate TAK-Net.
Clinical images, split lists, pretrained weights, checkpoints, and experiment logs are
not redistributed.

## Installation

The paper experiments used Python 3.10, PyTorch 2.6.0, and torchvision 0.21.0.

```bash
cd TAK-Net
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Download the ImageNet-21k pretrained hybrid R50-ViT-B/16 weights and place them at:

```text
weights/imagenet21k_R50+ViT-B_16.npz
```

The weights are distributed by the original ViT/TransUNet projects and are not included
in this repository. They can be downloaded with:

```bash
mkdir -p weights
curl -L \
  https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz \
  -o weights/imagenet21k_R50+ViT-B_16.npz
```

## Data layout

The paired images and masks must share the same filename:

```text
data/
  imgs/       # grayscale B-mode images
  ceus/       # spatially corresponding grayscale CEUS images
  masks/      # binary carotid-wall masks
```

Images may retain their original resolution; they are resized to 224 x 224 for model
input, and predictions are restored to the original image size. Mask values are `0` for
background and `255` for carotid wall. The optional value `128` marks ignored pixels.

Prepare the five-fold text files according to [splits/README.md](splits/README.md).
The data split must be case-disjoint.

## Configuration

The paper configuration is provided in
[`configs/experiment.json`](configs/experiment.json). Paths are resolved relative to the
repository root. If a ruler-scale CSV is available, set `ruler_scale_csv` to its path to
report HD95, ASSD, and wall-thickness profile MAE in millimetres. Otherwise, pixel-based
metrics are still reported. The scale CSV accepts the columns `relative_path`, `modality`,
and `px_per_cm`; only `relative_path` (or `file`) and `px_per_cm` are required.

## Training and evaluation

Run one fold:

```bash
taknet-train --config configs/experiment.json --folds 1
```

Run all five folds sequentially:

```bash
taknet-train --config configs/experiment.json
```

Evaluate existing checkpoints without retraining:

```bash
taknet-train --config configs/experiment.json --evaluate-only
```

Predicted masks, checkpoints, logs, and metric summaries are written
under `outputs/taknet_5fold/` by default.

To evaluate externally generated binary masks:

```bash
taknet-evaluate \
  --gt-dir data \
  --pred-dir path/to/pred_masks \
  --file-list splits/fold_1/test_files.txt \
  --output-dir outputs/external_eval \
  --run-name external
```

## Reproducibility notes

- The default random seed is 42, with deterministic PyTorch operations enabled when available.
- B-mode and CEUS share geometric augmentation parameters; intensity augmentation is sampled independently.
- The supplied configuration records all model, loss, optimization, and five-fold settings used by the main experiment.
- The clinical dataset is not included because it contains sensitive medical data.

## Acknowledgements

The hybrid ResNetV2-ViT backbone and decoder are adapted from
[TransUNet](https://github.com/Beckschen/TransUNet), which is distributed under
the Apache License 2.0. TAK-Net adds the paired B-mode/CEUS encoders, DAF, MCE,
LGD, task-specific training objectives, data pipeline, and evaluation code.

## License

This project is released under the Apache License 2.0.

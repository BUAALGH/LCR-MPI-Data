# T4FNet

This directory contains the standalone code for the T4FNet model used in the LCR-MPI experiments.

## Model

T4FNet uses two normalized low-concentration MPI channels:

- Channel 0: inverted third harmonic, `-L3`.
- Channel 1: fifth harmonic, `L5`.

The network contains:

- one Chebyshev mapping block from L3 to L5 features;
- SE fusion of measured and mapped L5 features;
- a frozen MPI-tuned MedSAM ViT-B mask generator;
- 4 x 4 patch embedding with 256 tokens;
- two Transformer stages with two blocks per stage;

The MedSAM-LoRA configuration is rank 4, alpha 8, dropout 0.1, and target module `qkv`. T4FNet freezes the complete mask generator while training the reconstruction network.

## Training configuration

| Setting               |                                 Value |
| --------------------- | ------------------------------------: |
| Cross-validation      |                               5 folds |
| Split per fold        | 65 train / 15 validation / 20 ID test |
| Random seed           |                                    42 |
| Epochs                |                                   100 |
| Batch size            |                               8 total |
| Optimizer             |                                 AdamW |
| Learning rate         |                                  1e-4 |
| Weight decay          |                                  1e-4 |
| AdamW betas / epsilon |                      PyTorch defaults |
| Scheduler             |                     CosineAnnealingLR |
| Minimum learning rate |                                  1e-7 |
| Scheduler update      |                        once per epoch |
| Reconstruction loss   |               0.5 L1 + 0.5 (1 - SSIM) |
| Mixed precision       |                       enabled on CUDA |
| Model selection       | lowest validation reconstruction loss |

## Data

Use the public `LCR-MPI-Data` directory. Zenodo: 10.5281/zenodo.22112372

## Environment

The recorded environment is Python 3.10.18, PyTorch 2.10.0, and CUDA 12.8. Install dependencies from the project directory:

```bash
conda env create -f environment.yml
conda activate t4fnet
```

The installed PyTorch build must match the local CUDA driver. If needed, install the correct PyTorch build first and then install the remaining packages.

## Checkpoints

Obtain the MedSAM ViT-B checkpoint from the official MedSAM project. Provide the MPI-tuned MedSAM-LoRA checkpoint separately. See `checkpoints/README.md`.

For the current workspace, the default configuration points to:

```text
../MedSAM_LoRA/medsam_vit_b.pth
../MedSAM_LoRA/results/mpiseg_medlora_20260706_205023/best_model.pth
```

Edit `configs/t4fnet_v4_b.json` or override these paths on the command line.

## Train

Run all five folds:

```bash
cd T4FNet-OpenSource
python train.py --config configs/t4fnet_v4_b.json
```

Run one fold:

```bash
python train.py --config configs/t4fnet_v4_b.json --fold 1
```

Training writes the full configuration, epoch history, latest checkpoint, best model, and test summary under `outputs/`.

## Inference

Run inference on the generalization data:

Use `--subset invivo` for the two in-vivo inputs. Predictions are stored as 64 x 64 NumPy arrays.

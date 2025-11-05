# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

T2I-Adapter is a lightweight (~70M parameters) network that provides extra guidance to pre-trained text-to-image diffusion models while freezing the original large models. The project implements various adapters for different control signals including sketch, depth, canny, color, style, openpose, keypose, and segmentation.

## Architecture

### Core Components

- **ldm/**: Main Latent Diffusion Models implementation
  - `models/diffusion/`: Diffusion sampling algorithms (DDIM, PLMS, DPM-Solver)
  - `modules/encoders/adapter.py`: Core Adapter network implementation
  - `data/dataset_*.py`: Dataset loaders for different adapter types
  - `inference_base.py`: Base inference utilities and argument parsers

- **configs/**: YAML configuration files for training and inference
  - `stable-diffusion/`: Stable Diffusion model configurations
  - `pl_train/`: PyTorch Lightning training configurations

- **Training Scripts**: `train_*.py` files for different adapter types
- **Testing Scripts**: `test_*.py` files for single and composable adapter inference

### Adapter Architecture

Adapters follow a consistent structure:
- Input: Condition image (depth, sketch, canny, etc.)
- Processing: Multi-scale feature extraction with residual blocks
- Output: Features injected into Stable Diffusion UNet layers
- Parameters: ~70M (300MB storage), trainable while SD model remains frozen

### CoAdapter Extension

CoAdapter (`docs/coadapter.md`) enables composable control by jointly training multiple adapters with an extra fuser network, allowing better synergy between different conditions.

## Common Development Commands

### Environment Setup
```bash
pip install -r requirements.txt
```

### Download Examples (Optional)
```bash
python examples/download_examples.py
```

### Training Commands

#### Depth Adapter Training
```bash
python train_depth.py \
  --config configs/stable-diffusion/sd-v1-train.yaml \
  --instance_name depth_experiment \
  --bsize 6 --epochs 150 --val_iter 2000 \
  --use_wandb --wandb_project t2i-adapter
```

#### General Training Pattern
```bash
python train_{adapter_type}.py \
  --config configs/stable-diffusion/sd-v1-train.yaml \
  --instance_name {experiment_name} \
  --auto_resume
```

### Inference Commands

#### Single Adapter Testing
```bash
# Depth adapter
python test_adapter.py --which_cond depth \
  --cond_path examples/depth/sd.png \
  --prompt "Stormtrooper's lecture, best quality" \
  --sd_ckpt models/v1-5-pruned-emaonly.ckpt \
  --adapter_ckpt models/t2iadapter_depth_sd14v1.pth

# Sketch adapter
python test_adapter.py --which_cond sketch \
  --cond_path examples/sketch/car.png \
  --prompt "A car with flying wings" \
  --sd_ckpt models/sd-v1-4.ckpt \
  --adapter_ckpt models/t2iadapter_sketch_sd14v1.pth
```

#### Composable Adapters
```bash
python test_composable_adapters.py \
  --prompt "1girl, computer desk, red chair best quality" \
  --depth_path examples/depth/desk_depth.png \
  --depth_adapter_ckpt experiments/train_depth/models/model_ad_70000.pth \
  --keypose_path examples/keypose/person_keypose.png \
  --keypose_adapter_ckpt models/t2iadapter_keypose_sd14v1.pth \
  --sd_ckpt models/anything-v4.5-pruned-fp16.ckpt
```

#### Gradio Demos
```bash
# CoAdapter demo
python app_coadapter.py --sd_ckpt models/v1-5-pruned-emaonly.ckpt

# Single adapter demo
python app.py --sd_ckpt models/v1-5-pruned-emaonly.ckpt
```

### Model Requirements

1. **Base SD Model**: Required Stable Diffusion checkpoint (v1.4/v1.5 recommended)
2. **Adapter Models**: Download from <https://huggingface.co/TencentARC/T2I-Adapter>
3. **Optional Models**: MMPose models for keypose detection

## Key Implementation Details

### Training Pipeline

- **Dataset Structure**: Each adapter type uses specific dataset format (see `ldm/data/dataset_*.py`)
- **Validation**: Uses fixed validation sample with comprehensive metrics (SSIM, PSNR, LPIPS, MSE, MAE)
- **Early Stopping**: Composite scoring system with configurable patience
- **Gradient Accumulation**: Supports effective batch size scaling
- **WandB Integration**: Automatic experiment tracking and logging

### Evaluation Metrics

The training code includes comprehensive image quality assessment:
- **SSIM**: Structural similarity (>0.5 is good)
- **PSNR**: Peak signal-to-noise ratio (>30dB is good)
- **LPIPS**: Perceptual similarity (<0.2 is good)
- **MSE/MAE**: Pixel-level errors (lower is better)
- **Composite Score**: Weighted combination for model selection

### Configuration System

- **Training Configs**: Define model architecture, learning rates, and training parameters
- **Inference Configs**: Specify sampling parameters and model paths
- **Adapter-Specific**: Each adapter type has dedicated configurations

### File Organization

- `experiments/{instance_name}/`: Training outputs organized by experiment name
  - `models/`: Training checkpoints
  - `result_ckpt/`: Best models (model_ad_best.pth)
  - `visualization/`: Generated samples and validation images
  - `training_states/`: Optimizer and training state files

## Development Notes

- The codebase supports both single-GPU and distributed training
- Adapter models are lightweight and designed for plug-and-play usage
- All training scripts include automatic resumption from checkpoints
- The inference system supports multiple sampling algorithms (DDIM, PLMS, DPM-Solver)
- WandB integration is optional but recommended for experiment tracking
## requirements
- 回答尽量采用中文，如果无法回答则回答英文。
# Image Reconstruction on ImageNet-32

Train an Energy Transformer to reconstruct images from partially masked inputs on ImageNet-32.

## Task

Image reconstruction on 32×32 downsampled ImageNet with 1000 classes. The model must learn representations that generalize across vastly different object categories.

## Dataset

- **Images**: 1,281,167 training, 50,000 validation
- **Size**: 32×32 pixels, RGB  
- **Classes**: 1,000 object categories
- **Source**: [Hugging Face](https://huggingface.co/datasets/benjamin-paine/imagenet-1k-32x32)
- **Patches**: 4×4 pixels, 64 patches per image
- **Normalization**: Mean=(0.485, 0.456, 0.406), Std=(0.229, 0.224, 0.225)

## Metrics

- **MSE**: Typically 0.05-0.06 (higher due to diversity)
- **PSNR**: Typically 24-27 dB
- **SSIM**: Typically 0.72-0.78

## Usage

### Training
```bash
guix shell -m manifest.scm -- python train.py
```

The dataset downloads automatically on first run (~636 MB).

### Visualization
```bash
python visualize.py              # Grid of reconstructions
python demo.py                   # Demo with multiple mask ratios
python infer.py --image random   # Single image inference
```

### Scaling Experiments
```bash
python scaling-data.py    # Test performance vs. dataset size
python scaling-model.py   # Test performance vs. model size
```

## Configuration

Key settings in `config.yaml`:
- `data.batch_size`: 256 (for GPU memory efficiency)
- `data.num_workers`: 10 (for faster data loading)
- `train.epochs`: 400 (sufficient for convergence)
- `train.optimizer.lr`: 3e-4 (scaled for batch size)


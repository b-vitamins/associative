# Image Reconstruction on CIFAR-10

Train an Energy Transformer to reconstruct images from partially masked inputs on CIFAR-10.

## Task

The model learns to reconstruct complete 32×32 images when 85% of patches are masked. Only the masked patches contribute to the training loss, forcing the model to learn meaningful representations.

## Dataset

- **Images**: 50,000 training, 10,000 validation
- **Size**: 32×32 pixels, RGB
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Patches**: 4×4 pixels, 64 patches per image
- **Normalization**: Mean=(0.4914, 0.4822, 0.4465), Std=(0.2470, 0.2435, 0.2616)

## Metrics

- **MSE**: Mean Squared Error, primary loss (lower is better)
- **PSNR**: Peak Signal-to-Noise Ratio in dB (higher is better, typically 25-30 dB)
- **SSIM**: Structural Similarity Index [0-1] (higher is better, typically 0.75-0.85)

## Usage

### Training
```bash
python train.py
```

Monitor real-time progress:
```
Epoch 100/1000 | Batch 391/391 | MSE: 0.023451 | PSNR: 28.3dB | SSIM: 0.8234 | LR: 3e-05 | t: 45s
```

### Visualization
```bash
python visualize.py              # Grid of reconstructions
python demo.py                   # Interactive demo with different mask ratios
python infer.py --image random   # Single image inference
```

### Configuration

Key parameters in `config.yaml`:
- `data.mask_ratio`: Fraction of patches to mask (default: 0.85)
- `model.embed_dim`: Hidden dimension (default: 256)
- `train.epochs`: Training epochs (default: 1000)

Override via command line:
```bash
python train.py data.mask_ratio=0.75 train.epochs=500
```

## Output

Results saved to `outputs/cifar10/YYYY-MM-DD/HH-MM-SS/`:
- `checkpoints/`: Model weights
- `train.log`: Training logs
- Visualizations from inference scripts
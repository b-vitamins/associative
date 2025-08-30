# Image Reconstruction on CIFAR-100

Train an Energy Transformer to reconstruct images from partially masked inputs on CIFAR-100.

## Task

Same as CIFAR-10 but with 100 fine-grained classes, making reconstruction more challenging due to increased visual diversity.

## Dataset

- **Images**: 50,000 training, 10,000 validation  
- **Size**: 32×32 pixels, RGB
- **Classes**: 100 fine-grained categories in 20 superclasses
- **Patches**: 4×4 pixels, 64 patches per image
- **Normalization**: Mean=(0.5071, 0.4866, 0.4409), Std=(0.2673, 0.2564, 0.2762)

## Metrics

- **MSE**: Typically 0.03-0.04 (slightly higher than CIFAR-10)
- **PSNR**: Typically 26-28 dB
- **SSIM**: Typically 0.78-0.82

## Usage

### Training
```bash
python train.py
```

### Visualization
```bash
python visualize.py              # Grid of reconstructions
python demo.py                   # Interactive demo
python infer.py --image random   # Single image inference
```

### Configuration

Same architecture as CIFAR-10 in `config.yaml`. The model handles the increased complexity without architectural changes.

## Expected Performance

Due to 10x more classes, expect:
- ~40% higher MSE than CIFAR-10
- ~2 dB lower PSNR
- Longer convergence time
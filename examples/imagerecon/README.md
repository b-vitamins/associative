# Image Reconstruction with Energy Transformers

This directory contains implementations of Image Reconstruction using Energy Transformers for self-supervised learning across multiple datasets.

## Task Definition

**Image Reconstruction** is a Computer Vision task that involves recovering complete images from partial, corrupted, or masked input data, enabling self-supervised representation learning. The task operates by randomly masking large portions of input images (typically 75-85% of patches) and training models to reconstruct the missing content, enabling the learning of rich visual representations without labeled data.

Contemporary methods leverage Vision Transformer architectures with asymmetric encoder-decoder designs, where lightweight encoders process only visible patches while specialized decoders reconstruct entire images from latent representations and positional encodings. This approach has revolutionized computer vision by adapting masked modeling to visual domains, achieving state-of-the-art performance in downstream tasks while requiring significantly less computational resources than traditional supervised pre-training.

## Technical Approach

### Architecture Overview

The Energy Transformer Image Reconstruction implementation uses:

- **Patch-based Processing**: Images are divided into non-overlapping patches (4x4 for CIFAR datasets)
- **Asymmetric Encoder-Decoder**: Lightweight encoder processes only visible patches, decoder reconstructs full images
- **Energy-based Dynamics**: Leverages energy minimization principles for stable training and reconstruction
- **High Masking Ratio**: 85% of patches are masked, forcing the model to learn meaningful representations

### Key Components

1. **Patch Embedding**: Converts image patches to token embeddings
2. **Mask Generation**: Random sampling of patches to mask during training
3. **Energy Transformer**: Processes visible patches through attention mechanisms
4. **Reconstruction Head**: Predicts pixel values for masked patches
5. **Loss Function**: Mean Squared Error on masked patches only

## Applications

The Image Reconstruction approach implemented here enables:

- **Self-supervised Visual Representation Learning**: Pre-training for computer vision without labels
- **Medical Image Denoising and Reconstruction**: Diagnostic imaging enhancement
- **Image Inpainting and Completion**: Restoring damaged or corrupted visual content
- **Computational Photography**: Image enhancement and restoration applications
- **Video Compression and Restoration**: Temporal masked reconstruction techniques
- **Data Augmentation**: Improved generalization through image reconstruction
- **Privacy-preserving Computer Vision**: Learning from partially obscured data
- **Industrial Quality Control**: Automated defect detection and reconstruction
- **Satellite and Remote Sensing**: Earth observation image enhancement

## Datasets

### CIFAR-10 (`cifar-10/`)

**Structure**: 60,000 32x32 color images in 10 classes
- **Training**: 50,000 images
- **Test**: 10,000 images
- **Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Patch Configuration**: 4x4 patches, 64 total patches per image
- **Use Case**: Rapid prototyping and algorithm testing with sufficient complexity

The small image size and limited resolution make CIFAR-10 ideal for evaluating reconstruction quality and representation learning capabilities. The balanced class distribution and well-defined splits provide reliable benchmarking for image reconstruction approaches.

### CIFAR-100 (`cifar-100/`)

**Structure**: 60,000 32x32 color images in 100 fine-grained classes
- **Training**: 50,000 images  
- **Test**: 10,000 images
- **Classes**: 100 fine-grained categories grouped into 20 superclasses
- **Patch Configuration**: 4x4 patches, 64 total patches per image
- **Challenge**: Higher diversity and complexity than CIFAR-10

CIFAR-100 provides increased complexity with more fine-grained categories, making it a more challenging benchmark for representation learning. The dataset tests the model's ability to capture subtle visual differences and learn hierarchical representations.


## Evaluation Metrics

### Primary Metrics

1. **MSE (Mean Squared Error)**
   - **Range**: [0, ∞], optimal: 0
   - **Purpose**: Primary training and validation loss
   - **Application**: Pixel-level reconstruction accuracy measurement
   - **Characteristics**: Per-sample metric, aggregated by mean

2. **PSNR (Peak Signal-to-Noise Ratio)**
   - **Range**: [0, ∞], optimal: max (typically 20-50 dB)
   - **Unit**: decibels (dB)
   - **Purpose**: Signal quality assessment for reconstructed images
   - **Application**: Standard image quality metric in computer vision benchmarks
   - **Limitations**: May not align with perceptual quality

3. **SSIM (Structural Similarity Index)**
   - **Range**: [0, 1], optimal: 1
   - **Purpose**: Perceptual similarity measurement capturing luminance, contrast, and structure
   - **Application**: Human visual system aligned quality assessment
   - **Advantages**: Better correlation with perceived quality than MSE/PSNR

### Advanced Metrics

4. **LPIPS (Learned Perceptual Image Patch Similarity)**
   - **Range**: [0, 10.0], optimal: 0
   - **Purpose**: Deep learning based perceptual distance measurement
   - **Application**: Perceptual quality assessment using neural network features
   - **Characteristics**: Computationally expensive, requires pre-trained networks
   - **Limitations**: Dependent on network architecture choice

5. **FID (Frechet Inception Distance)**
   - **Range**: [0, 1000.0], optimal: 0
   - **Purpose**: Distributional similarity between real and reconstructed images
   - **Application**: Generative model evaluation and dataset-wide quality assessment
   - **Level**: Dataset-wide metric comparing feature statistics
   - **Limitations**: Requires large sample sizes for stable estimates

## Model Architecture Details

### Energy Transformer Configuration

The implementation uses consistent architecture across datasets with adaptations for image dimensions:

- **Embedding Dimension**: 256
- **Attention Heads**: 12
- **MLP Ratio**: 4.0
- **Number of Layers**: 1 (lightweight encoder)
- **Energy Time Steps**: 12
- **Step Size (α)**: 10.0
- **Attention Beta**: 0.125

### Training Configuration

- **Optimizer**: AdamW with β₁=0.9, β₂=0.999
- **Learning Rate**: 5e-5 with cosine annealing to 1e-6
- **Weight Decay**: 1e-4
- **Batch Size**: 128
- **Epochs**: 1000
- **Gradient Clipping**: 1.0
- **Mask Ratio**: 85% (following best practices for masked image reconstruction)

## State-of-the-Art Models

### Contemporary Approaches

1. **Masked Autoencoder (MAE)** - O64
   - **Architecture**: Vision Transformer
   - **Year**: 2021
   - **Key Innovation**: High masking ratio (75%) with asymmetric encoder-decoder
   - **Capabilities**: Self-supervised pre-training, transfer learning, fine-tuning adaptation

2. **BEiT (BERT pre-training of Image Transformers)** - O118  
   - **Architecture**: Vision Transformer
   - **Year**: 2021
   - **Key Innovation**: Discrete visual token prediction using dVAE
   - **Capabilities**: Masked image modeling, downstream task adaptation, multi-scale processing

3. **MixedAE (Mixed Autoencoder)** - O119
   - **Architecture**: Hybrid Vision Transformer
   - **Year**: 2023
   - **Key Innovation**: Mixed reconstruction objectives combining pixel and token prediction
   - **Capabilities**: Enhanced representation learning, improved fine-tuning performance, robust feature extraction

### Supporting Architecture

4. **Swin Transformer** - O31
   - **Role**: Alternative backbone for hierarchical feature learning
   - **Capabilities**: Multi-scale attention, efficient computation, strong performance

## Implementation Structure

### Directory Organization

```
imagerecon/
├── cifar-10/          # CIFAR-10 specific implementation
│   ├── config.yaml    # Model and training configuration
│   ├── train.py       # Training script
│   ├── visualize.py   # Reconstruction visualization
│   └── demo.py        # Interactive demonstration
├── cifar-100/         # CIFAR-100 implementation
│   ├── config.yaml    # Adapted configuration
│   ├── train.py       # Training with CIFAR-100 specifics
│   └── visualize.py   # Visualization tools
└── README.md          # This documentation
```

### Key Features

- **Hydra Configuration**: Flexible parameter management and experiment tracking
- **Comprehensive Logging**: Training metrics, validation scores, and timing information
- **Visualization Tools**: Reconstruction quality assessment and training progress
- **Checkpointing**: Best model saving and training resumption support
- **GPU Acceleration**: CUDA support with fallback to CPU

## Usage Instructions

### Training

```bash
# CIFAR-10 training
cd cifar-10/
python train.py

# CIFAR-100 training  
cd cifar-100/
python train.py

```

### Visualization

```bash
# Generate reconstruction visualizations
cd cifar-10/
python visualize.py

# View training progress (if available)
tensorboard --logdir outputs/
```

### Configuration

Modify `config.yaml` files to adjust:

- **Model Architecture**: embedding dimensions, attention heads, layers
- **Training Hyperparameters**: learning rate, batch size, epochs
- **Data Settings**: mask ratio, augmentation, dataset paths
- **Output Directories**: checkpoint and visualization locations

## Research Context

### Historical Development

Image Reconstruction with masking represents a paradigm shift from traditional supervised pre-training to self-supervised learning in computer vision. Inspired by the success of BERT in natural language processing, this approach adapts masked modeling to visual domains by treating images as sequences of patches and learning to predict masked content.

### Key Innovations

1. **High Masking Ratios**: Unlike NLP (15% masking), vision benefits from aggressive masking (75-85%)
2. **Asymmetric Architecture**: Encoder processes only visible patches, reducing computation
3. **Patch-based Tokenization**: Direct pixel prediction without discrete tokenization
4. **Energy-based Dynamics**: Leveraging energy minimization for stable convergence

### Impact and Applications

The Image Reconstruction approach has enabled significant advances in:

- **Transfer Learning**: Strong performance on downstream classification, detection, segmentation
- **Data Efficiency**: Reduced labeled data requirements through effective pre-training
- **Computational Efficiency**: Lower pre-training costs compared to supervised methods
- **Representation Quality**: Rich features learned through reconstruction objectives

This implementation provides a foundation for exploring these concepts and developing novel approaches to self-supervised visual representation learning.
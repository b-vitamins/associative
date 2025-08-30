# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyTorch implementation of associative memory models using energy-based dynamics, including Hopfield networks and energy transformers for vision and graph learning tasks.

## Development Commands

### Environment Setup
```bash
# Always use Guix shell for development
guix shell -m manifest.scm -- <command>

# Install dependencies (read-only - use Guix packages)
guix shell -m manifest.scm -- poetry install
```

### Testing & Quality
```bash
# Run all tests
guix shell -m manifest.scm -- pytest tests/

# Run specific test file
guix shell -m manifest.scm -- pytest tests/unit/nn/test_transformer.py -xvs

# Run with coverage
guix shell -m manifest.scm -- pytest tests/ --cov=associative --cov-report=term-missing

# Code quality checks
guix shell -m manifest.scm -- ruff check .
guix shell -m manifest.scm -- ruff format .
guix shell -m manifest.scm -- pyright
```

### Training Examples
```bash
# Vision tasks (Image Reconstruction)
cd examples/imagerecon/cifar-10
guix shell -m ../../../manifest.scm -- python train.py

# With custom config overrides (Hydra)
guix shell -m ../../../manifest.scm -- python train.py train.max_epochs=100 model.num_layers=2

# Graph classification tasks
cd examples/graphclass/mutag
guix shell -m ../../../manifest.scm -- python train.py
```

## Architecture

### Core Components

**Energy-Based Dynamics**: Models evolve hidden states through gradient descent on energy functions, implementing associative memory through iterative refinement.

**Model Classes**:
- `EnergyTransformer`: Vision transformer with energy-based attention for image tasks
- `GraphEnergyTransformer`: Graph transformer variant with adjacency-aware attention
- `EnergyTransformerBlock`: Core building block with energy attention and Hopfield layers

**Key Modules** (`associative/nn/modules/`):
- `attention.py`: Energy-based attention mechanism with gradient dynamics
- `hopfield.py`: Modern Hopfield layers with configurable energy functions
- `transformer.py`: Main transformer architectures for vision and graphs
- `config.py`: Dataclass configurations for all components

### Configuration System

Uses Hydra for hierarchical configuration:
- Base configs: `configs/` (model, data, training defaults)
- Task-specific: `examples/*/config.yaml` (overrides for each dataset/task)
- Command-line overrides supported via dot notation

### Dataset Support

**Vision**: CIFAR-10/100, ImageNet32 (custom downsampled version)
**Graph**: MUTAG, PROTEINS, DD, NCI, ZINC
**Multimodal**: GRID (audio-visual), MovieChat (video QA)

Custom datasets in `associative/datasets/` with PyTorch DataLoader interfaces.

## Key Implementation Details

### Energy Attention
- Computes attention weights through energy minimization
- Configurable temperature scaling (`attn_beta`)
- Supports masked attention for autoencoding tasks

### Hopfield Layers
- Multiple activation variants: ReLU, GELU, Manhattan, Softmax
- Energy function: `-0.5 * activation(Wx)^2.sum()` 
- Configurable hidden dimension ratio

### Graph Processing
- Uses PyTorch Geometric for data handling
- Positional encodings via Laplacian eigenvectors
- Adjacency matrix integration in attention

### Training Utilities
- Mask generation for masked autoencoding (`generate_mask_indices`)
- Reconstruction metrics (MSE, PSNR, SSIM)
- Cosine annealing learning rate scheduling

## Testing Strategy

- Unit tests: Individual module functionality (`tests/unit/`)
- Integration tests: End-to-end training (`tests/integration/`)
- Slow tests marked with `@pytest.mark.slow`
- Fixture-based testing with PyTorch tensors

## Ruff Configuration

Line length: 88, Target: Python 3.11
Selected rules: E, F, I, N, W, B, C90, UP, YTT, S, A, C4, RET, SIM, PD, PL, PERF, RUF
Ignored: E501 (line length), S101 (assert usage)
Per-file ignores for PLR0913 (too many arguments) in functional APIs
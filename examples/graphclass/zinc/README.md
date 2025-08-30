# Graph Classification on ZINC

This example demonstrates training a Graph Energy Transformer on the ZINC molecular property prediction dataset.

## Dataset

ZINC is a graph regression benchmark where the task is to predict the constrained solubility (logP - SA - cycles) of molecules. Each graph represents a molecule with:
- Nodes: Atoms (with 1D features)
- Edges: Chemical bonds
- Target: Molecular property (continuous value)

## Quick Start

### Training
```bash
python train.py
```

### Visualization
```bash
# Visualize embeddings and predictions
python visualize.py
```

## Configuration

The default configuration is optimized for ZINC:

- **Model**: 256-dim embeddings, 8 heads, 1 layer
- **Training**: 500 epochs, AdamW optimizer, MAE loss
- **Data**: Batch size 128, Laplacian positional encoding
- **Learning rate**: 1e-4 with cosine annealing

Modify `config.yaml` or use Hydra overrides:
```bash
python train.py model.num_heads=16 train.epochs=1000 model.embed_dim=512
```

## Outputs

- **Checkpoints**: `outputs/ZINC/*/checkpoints/`
- **Visualizations**: `visualizations/` (t-SNE, PCA, predictions)
- **Logs**: Hydra logs in `outputs/ZINC/*/`

## Performance

Expected results after 500 epochs:
- Test MAE: ~0.15-0.20 (lower is better)
- The model should learn meaningful molecular embeddings visible in t-SNE plots

## Notes

- ZINC is a regression task (not classification), so we use MAE loss
- The dataset has relatively small graphs (max 38 nodes)
- Positional encodings significantly improve performance on this dataset
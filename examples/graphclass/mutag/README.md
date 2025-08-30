# Graph Classification on MUTAG

This example demonstrates training a Graph Energy Transformer on the MUTAG dataset for molecular mutagenicity prediction.

## Dataset

MUTAG is a binary graph classification benchmark containing 188 molecular graphs. The task is to predict whether a molecule is mutagenic (can cause genetic mutations). Each graph represents:
- Nodes: Atoms with 7D features (atom type, etc.)
- Edges: Chemical bonds
- Target: Binary label (mutagenic or non-mutagenic)

## Quick Start

### Training
```bash
python train.py
```

### Visualization
```bash
# Visualize embeddings and performance metrics
python visualize.py
```

## Configuration

The default configuration is optimized for this small dataset:

- **Model**: 128-dim embeddings, 8 heads, 1 layer
- **Training**: 200 epochs, AdamW optimizer, cross-entropy loss
- **Data**: Batch size 32, 80/10/10 train/val/test split
- **Learning rate**: 5e-4 with cosine annealing

Modify `config.yaml` or use Hydra overrides:
```bash
python train.py model.embed_dim=256 train.epochs=300 model.num_heads=16
```

## Outputs

- **Checkpoints**: `outputs/MUTAG/*/checkpoints/`
- **Visualizations**: `visualizations/` containing:
  - Confusion matrix
  - t-SNE embedding visualization
  - Per-class performance metrics
- **Logs**: Hydra logs in `outputs/MUTAG/*/`

## Performance

Expected results after 200 epochs:
- Test accuracy: ~85-90%
- The model should learn to separate mutagenic and non-mutagenic molecules
- Clear clustering should be visible in t-SNE plots

## Notes

- MUTAG is a small dataset (188 graphs), so careful regularization is needed
- Node features are important for this task
- Consider using cross-validation for more robust evaluation
- The dataset is imbalanced (~60% positive class)
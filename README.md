# Associative

PyTorch implementation of associative memory models including Hopfield networks and energy-based transformers for vision and graph tasks.

## Installation

### Development Setup with Guix

```bash
git clone https://github.com/b-vitamins/associative.git
cd associative
guix shell -m manifest.scm
poetry install
```

### Alternative Installation

```bash
pip install -e .
```

## Usage

### Basic Example

```python
import torch
from associative import EnergyTransformer, EnergyTransformerConfig

# Create configuration
config = EnergyTransformerConfig(
    patch_size=4,
    num_patches=64,  # For 32x32 images
    embed_dim=256,
    num_layers=1,
    num_heads=12,
    num_time_steps=12
)

# Initialize model
model = EnergyTransformer(config)

# Forward pass
images = torch.randn(4, 3, 32, 32)
output = model(images)  # Shape: [4, 64, 48]
```

### Masked Autoencoding

```python
# Generate mask indices
batch_idx = torch.arange(4).unsqueeze(1)
mask_idx = torch.randint(0, 64, (4, 32))

# Forward with mask
output = model(images, mask=(batch_idx, mask_idx))
```

### Graph Classification

```python
from associative import GraphEnergyTransformer
from associative.utils.graph import prepare_graph_batch
from torch_geometric.data import Batch, Data

# Create graph model
config = EnergyTransformerConfig(
    input_dim=32,
    embed_dim=128,
    num_layers=2,
    pos_encoding_dim=10,
    out_dim=2  # Binary classification
)
model = GraphEnergyTransformer(config)

# Prepare graph data
data = Data(x=torch.randn(5, 32), edge_index=torch.tensor([[0,1,2], [1,2,0]]))
batch = Batch.from_data_list([data])

# Process batch
x, adj, _, pos_enc, mask = prepare_graph_batch(batch)
output = model(x[:, 1:], adj[:, 1:, 1:], pos_enc, mask[:, 1:])
graph_embedding = output['graph_embedding']
```

### Training Examples

Vision tasks (Masked Autoencoding):
```bash
cd examples/mae/cifar-10
python train.py

# Or with custom config
python train.py train.max_epochs=100
```

Graph classification tasks:
```bash
cd examples/graphclass/mutag
python train.py

# Other datasets
cd examples/graphclass/proteins
python train.py
```

## Architecture

The associative memory models use gradient-based dynamics to evolve hidden states by minimizing an energy function. Key components:

- **Energy Attention**: Attention mechanism based on energy minimization
- **Hopfield Layers**: Energy-based feed-forward networks with configurable variants
- **Gradient Dynamics**: Iterative refinement through gradient descent
- **Graph Support**: Adjacency-aware attention and graph positional encodings

### Configuration

The model is highly configurable through YAML files:
- Base configs in `configs/`
- Example-specific configs in `examples/*/config.yaml`
- Supports Hydra for command-line overrides

### Examples Structure

- `examples/mae/`: Masked autoencoding for CIFAR-10/100
- `examples/graphclass/`: Graph classification for MUTAG, PROTEINS, DD, NCI, ZINC

## Testing

```bash
# Run all tests
pytest tests/

# Check code quality
ruff check .
ruff format .
pyright
```

## Development Tools

- **Linting**: ruff
- **Type checking**: pyright
- **Testing**: pytest
- **Build**: poetry

## License

MIT
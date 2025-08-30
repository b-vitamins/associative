# Graph Classification with Energy Transformers

This directory contains implementations for training Graph Energy Transformers on standard graph classification benchmarks. Graph classification is a fundamental machine learning task that assigns entire graphs to predefined classes based on their structural properties, node features, and edge relationships.

## Table of Contents

- [Task Definition](#task-definition)
- [Applications](#applications)  
- [Datasets](#datasets)
- [Models & Approaches](#models--approaches)
- [Evaluation Metrics](#evaluation-metrics)
- [Implementation Details](#implementation-details)
- [Usage](#usage)
- [Performance Results](#performance-results)

## Task Definition

**Graph Classification** is a Graph Learning task that operates at the graph level, requiring models to learn representations that capture both local connectivity patterns and global topological characteristics across diverse graph structures. Unlike node-level tasks that predict properties of individual nodes, graph classification assigns labels to entire graphs based on their overall structural and semantic properties.

### Key Characteristics

- **Input**: Complete graphs with node features, edges, and optional edge attributes
- **Output**: Class labels for entire graphs (binary or multi-class)
- **Challenge**: Learning graph-level representations that capture both local patterns and global topology
- **Evaluation**: Typically uses 10-fold cross-validation with accuracy as the primary metric

## Applications

Graph classification has diverse real-world applications across multiple domains:

1. **Molecular Property Prediction**: Drug discovery and toxicity assessment of chemical compounds
2. **Social Network Analysis**: Community detection and user behavior modeling
3. **Protein Function Prediction**: Bioinformatics and structural biology applications
4. **Fraud Detection**: Financial transaction networks and payment systems
5. **Recommendation Systems**: Social graphs and user interaction networks
6. **Malware Detection**: Program call graphs and behavioral analysis
7. **Material Property Prediction**: Chemistry and materials science applications
8. **Biological Network Analysis**: Disease pathway identification and biomarker discovery

## Datasets

This implementation includes four standard benchmark datasets from the TUDataset collection:

### MUTAG (Molecular Mutagenicity)
- **Size**: 188 molecular graphs
- **Classes**: 2 (mutagenic vs non-mutagenic)
- **Domain**: Chemistry - nitroaromatic compounds
- **Nodes**: Atoms (7 discrete labels for atom types)
- **Edges**: Chemical bonds (single, double, triple, aromatic)
- **Average**: ~18 nodes, ~20 edges per graph
- **Task**: Predict mutagenicity on Salmonella typhimurium
- **Origin**: Structure-activity relationship research (Debnath et al., 1991)

### PROTEINS (Enzyme Classification)
- **Size**: 1,113 protein graphs  
- **Classes**: 2 (enzyme vs non-enzyme)
- **Domain**: Bioinformatics - protein structures
- **Nodes**: Amino acids with structural properties
- **Edges**: Spatial proximity (< 6 Angstroms) or sequence neighbors
- **Average**: ~39 nodes, ~73 edges per graph
- **Task**: Distinguish enzymatic from non-enzymatic proteins
- **Features**: Secondary structure + physicochemical properties

### NCI1 (Anti-Cancer Screening)
- **Size**: 4,110+ chemical compounds
- **Classes**: 2 (active vs inactive against lung cancer)
- **Domain**: Chemistry - anti-cancer drug screening
- **Nodes**: Atoms with one-hot encoded atom types
- **Edges**: Chemical bonds between atoms
- **Task**: Predict activity against non-small cell lung cancer
- **Origin**: National Cancer Institute screening program

### DD (Dobson & Doig Proteins)
- **Size**: 1,178 protein structures
- **Classes**: 2 (enzyme vs non-enzyme)
- **Domain**: Bioinformatics - larger protein structures
- **Nodes**: Amino acids (89-dimensional one-hot labels)
- **Edges**: Spatial and sequential relationships
- **Task**: Binary protein function classification
- **Characteristics**: Larger and more complex than PROTEINS dataset

## Models & Approaches

### Graph Neural Network Architectures

Modern graph classification leverages various Graph Neural Network (GNN) architectures:

#### Graph Convolutional Networks (GCN)
- **Year**: 2017
- **Approach**: Spectral convolution with localized first-order approximation
- **Strengths**: Efficient message passing, strong theoretical foundation
- **Limitations**: Over-smoothing with depth, sensitive to graph structure

#### GraphSAGE (Graph Sample and Aggregate)  
- **Year**: 2017
- **Approach**: Inductive learning via neighborhood sampling and aggregation
- **Strengths**: Scalable, handles unseen nodes, multiple aggregation functions
- **Limitations**: Sampling may miss long-range dependencies

#### Graph Attention Networks (GAT)
- **Year**: 2018  
- **Approach**: Self-attention mechanisms for adaptive neighborhood weighting
- **Strengths**: Interpretable attention weights, handles varying node degrees
- **Limitations**: Quadratic complexity, memory intensive for dense graphs

#### Graph Isomorphism Networks (GIN)
- **Year**: 2019
- **Approach**: Theoretically grounded matching Weisfeiler-Lehman test power
- **Strengths**: Maximum expressive power among message-passing networks
- **Limitations**: Higher computational cost, sensitive to pooling strategies

### Energy Transformer Approach

This implementation uses **Graph Energy Transformers**, which combine:
- **Energy-based dynamics**: Iterative refinement through energy minimization
- **Attention mechanisms**: Multi-head self-attention for node interactions
- **Positional encodings**: Eigen-based encodings to capture graph structure
- **Graph pooling**: CLS token approach for graph-level representations

## Evaluation Metrics

### Primary Metrics

#### Accuracy  
- **Range**: 0-100%
- **Optimal**: Higher is better
- **Usage**: Primary metric for all datasets, 10-fold cross-validation
- **Description**: Percentage of correctly classified graphs

#### F1-Score
- **Range**: 0-1  
- **Optimal**: Higher is better
- **Usage**: Harmonic mean of precision and recall
- **Application**: Particularly important for imbalanced datasets

#### ROC-AUC (Area Under ROC Curve)
- **Range**: 0-1
- **Optimal**: Higher is better  
- **Usage**: Binary classification evaluation across decision thresholds
- **Advantages**: Threshold-independent, good for imbalanced data
- **Limitations**: Only applicable to binary classification

### Evaluation Protocol

Standard evaluation follows these practices:
- **Cross-validation**: 10-fold CV for robust performance estimation
- **Data splits**: 80/10/10 train/validation/test for development
- **Metrics reporting**: Mean ± standard deviation across folds
- **Statistical significance**: Paired t-tests for model comparisons

## Implementation Details

### Architecture Configuration

Each dataset requires specific configuration due to varying characteristics:

#### MUTAG Configuration
```yaml
data:
  max_num_nodes: 28
  batch_size: 16
model:
  input_dim: 12  # 7D node features + 5D degree features
  embed_dim: 128
  num_heads: 8
  num_layers: 1
train:
  epochs: 300
  lr: 1e-3
```

#### PROTEINS Configuration  
```yaml
data:
  max_num_nodes: 620
  batch_size: 8
model:
  input_dim: 4   # 4D node features
  embed_dim: 64
  num_heads: 4
  num_layers: 1
train:
  epochs: 300
  lr: 3e-4
```

#### NCI1 Configuration
```yaml
data:
  max_num_nodes: 111
  batch_size: 64
model:
  input_dim: 37  # 37 unique node labels
  embed_dim: 256
  num_heads: 8
  num_layers: 1
train:
  epochs: 200
  lr: 3e-4
```

#### DD Configuration
```yaml
data:
  max_num_nodes: 300
  batch_size: 2
model:
  input_dim: 89  # 89D one-hot node labels
  embed_dim: 32
  num_heads: 2
  num_layers: 1
train:
  epochs: 200
  lr: 3e-4
```

### Training Process

1. **Data Loading**: PyTorch Geometric TUDataset with preprocessing
2. **Feature Engineering**: Node degree features, positional encodings
3. **Batch Preparation**: Graph batching with padding to max_num_nodes
4. **Forward Pass**: Energy transformer with iterative refinement
5. **Loss Computation**: Cross-entropy loss for classification
6. **Optimization**: AdamW optimizer with gradient clipping
7. **Evaluation**: Validation and test set evaluation per epoch

### Key Implementation Features

- **Memory Management**: Careful batch sizing for large graphs (DD, PROTEINS)
- **Gradient Accumulation**: Used for datasets with small batch sizes
- **Positional Encodings**: Eigen-decomposition based structural encodings
- **Regularization**: Dropout, weight decay, gradient clipping
- **Checkpointing**: Best model saving based on validation accuracy

## Usage

### Training a Model

Navigate to the specific dataset directory and run:

```bash
# MUTAG
cd mutag/
python train.py

# PROTEINS  
cd proteins/
python train.py

# NCI1
cd nci/
python train.py

# DD
cd dd/
python train.py
```

### Configuration Overrides

Use Hydra to modify configurations:

```bash
# Increase model size
python train.py model.embed_dim=256 model.num_heads=16

# Adjust training
python train.py train.epochs=500 train.optimizer.lr=1e-4

# Change batch size
python train.py data.batch_size=32
```

### Visualization

Generate visualization plots after training:

```bash
python visualize.py
```

This creates:
- Confusion matrices
- t-SNE embedding visualizations  
- Per-class performance metrics
- Training curves

## Performance Results

### Expected Performance Ranges

Based on literature and empirical results:

| Dataset  | Expected Accuracy | Energy Transformer | Notes |
|----------|-------------------|-------------------|-------|
| MUTAG    | 85-90%           | ~87-89%           | Small dataset, high variance |
| PROTEINS | 75-80%           | ~76-78%           | Moderate size, balanced |
| NCI1     | 80-85%           | ~82-84%           | Large dataset, stable |
| DD       | 75-80%           | ~77-79%           | Large graphs, memory intensive |

### State-of-the-Art Comparisons

Current SOTA results from literature:

- **MUTAG**: 89.4% (GIN)
- **PROTEINS**: 78.1% (GraphSAGE)  
- **NCI1**: 84.2% (GCN)
- **DD**: 79.3% (GAT)

### Training Characteristics

- **MUTAG**: Fast training (~5-10 minutes), high variance due to small size
- **PROTEINS**: Moderate training time (~20-30 minutes), memory intensive
- **NCI1**: Longer training (~30-45 minutes), stable convergence
- **DD**: Slowest training (~45-60 minutes), requires careful memory management

## Directory Structure

```
graphclass/
├── README.md              # This file
├── mutag/                 # MUTAG implementation
│   ├── config.yaml        # MUTAG-specific configuration
│   ├── train.py          # Training script
│   ├── visualize.py      # Visualization utilities
│   └── data/             # MUTAG dataset files
├── proteins/             # PROTEINS implementation  
│   ├── config.yaml       # PROTEINS-specific configuration
│   ├── train.py         # Training script
│   └── visualize.py     # Visualization utilities
├── nci/                 # NCI1 implementation
│   ├── config.yaml      # NCI1-specific configuration
│   ├── train.py        # Training script
│   └── visualize.py    # Visualization utilities
└── dd/                 # DD implementation
    ├── config.yaml     # DD-specific configuration
    ├── train.py       # Training script
    └── visualize.py   # Visualization utilities
```

## References

1. **MUTAG**: Debnath et al. "Structure-activity relationship of mutagenic aromatic and heteroaromatic nitro compounds" (1991)
2. **PROTEINS**: Borgwardt et al. "Protein function prediction via graph kernels" (2005)  
3. **NCI1**: Wale et al. "Comparison of descriptor spaces for chemical compound retrieval and classification" (2008)
4. **GCN**: Kipf & Welling "Semi-Supervised Classification with Graph Convolutional Networks" (2017)
5. **GraphSAGE**: Hamilton et al. "Inductive Representation Learning on Large Graphs" (2017)
6. **GAT**: Veličković et al. "Graph Attention Networks" (2018)
7. **GIN**: Xu et al. "How Powerful are Graph Neural Networks?" (2019)

This implementation provides a comprehensive framework for graph classification research and practical applications, combining theoretical rigor with empirical validation across standard benchmarks.
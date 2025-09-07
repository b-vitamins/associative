"""Neural network modules for associative memory models.

This module provides PyTorch implementations of energy-based associative memory
architectures, including modern Hopfield networks, energy transformers, and
continuous memory systems. Key components include:

- Attention mechanisms with energy-based dynamics
- Basis functions for continuous memory representations  
- Configuration classes for model parameterization
- Continuous memory modules with function approximation
- Hopfield networks with modern architectures
- Numerical integrators for continuous dynamics
- Energy-based normalization layers
- Optimization utilities for energy minimization
- Transformer architectures with associative memory
- Vision-specific components like patch embedding

All modules support GPU acceleration and are designed for scalability
in computer vision, graph learning, and multimodal tasks.
"""

from .attention import EnergyAttention, GraphEnergyAttention, MultimodalEnergyAttention
from .basis import (
    BasisFunction,
    ContinuousCompression,
    FourierBasis,
    GaussianBasis,
    PolynomialBasis,
    RectangularBasis,
    create_basis,
)
from .config import (
    BasisConfig,
    CCCPConfig,
    ContinuousHopfieldConfig,
    EnergyAttentionConfig,
    EnergyTransformerConfig,
    HopfieldConfig,
    METConfig,
)
from .continuous import ContinuousAttention, ContinuousHopfield, ContinuousMemory
from .hopfield import CrossModalHopfield, Hopfield
from .integrator import (
    AdaptiveIntegrator,
    GaussLegendreIntegrator,
    Integrator,
    MonteCarloIntegrator,
    SimpsonIntegrator,
    TrapezoidalIntegrator,
    create_integrator,
)
from .normalization import EnergyLayerNorm
from .optimization import (
    CCCPOptimizer,
    ContinuousHopfieldEnergy,
    EnergyFunction,
    OptimizationResult,
    QuadraticConvexSolver,
)
from .transformer import (
    EnergyTransformer,
    EnergyTransformerBlock,
    GraphEnergyBlock,
    GraphEnergyTransformer,
    METBlock,
    MultimodalEnergyTransformer,
)
from .utils import Lambda
from .vision import PatchEmbed

__all__ = [
    # Integration utilities
    "AdaptiveIntegrator",
    # Configuration classes
    "BasisConfig",
    # Basis functions
    "BasisFunction",
    "CCCPConfig",
    "CCCPOptimizer",
    # Continuous modules
    "ContinuousAttention",
    # Basis and compression
    "ContinuousCompression",
    "ContinuousHopfield",
    "ContinuousHopfieldConfig",
    "ContinuousHopfieldEnergy",
    "ContinuousMemory",
    # Hopfield modules
    "CrossModalHopfield",
    # Attention modules
    "EnergyAttention",
    "EnergyAttentionConfig",
    "EnergyFunction",
    # Normalization
    "EnergyLayerNorm",
    # Transformer modules
    "EnergyTransformer",
    "EnergyTransformerBlock",
    "EnergyTransformerConfig",
    "FourierBasis",
    "GaussLegendreIntegrator",
    "GaussianBasis",
    "GraphEnergyAttention",
    "GraphEnergyBlock",
    "GraphEnergyTransformer",
    "Hopfield",
    "HopfieldConfig",
    "Integrator",
    # Utilities
    "Lambda",
    "METBlock",
    "METConfig",
    "MonteCarloIntegrator",
    "MultimodalEnergyAttention",
    "MultimodalEnergyTransformer",
    "OptimizationResult",
    "PatchEmbed",
    "PolynomialBasis",
    "QuadraticConvexSolver",
    "RectangularBasis",
    "SimpsonIntegrator",
    "TrapezoidalIntegrator",
    "create_basis",
    "create_integrator",
]

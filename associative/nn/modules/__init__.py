"""Associative memory model modules."""

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

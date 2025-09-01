"""Registry utilities for video processing components.

This module provides registration patterns following PyTorch's approach
for registering optimizers, schedulers, etc. Allows for extensible
video processing components.
"""

# Global registries for different component types
EMBEDDER_REGISTRY: dict[str, type] = {}
TRANSFORM_REGISTRY: dict[str, type] = {}
LOADER_REGISTRY: dict[str, type] = {}
METRIC_REGISTRY: dict[str, type] = {}


def register_embedder(name: str, embedder_class: type) -> type:
    """Register an embedding extractor class.

    Args:
        name: Name to register under
        embedder_class: Class implementing EmbeddingExtractor interface

    Returns:
        The registered class (for use as decorator)

    Raises:
        ValueError: If name already registered with different class
        TypeError: If embedder_class doesn't have required interface

    Example:
        >>> @register_embedder("my_embedder")
        ... class MyEmbedder(EmbeddingExtractor):
        ...     pass

        >>> # Or directly
        >>> register_embedder("my_embedder", MyEmbedder)
    """
    if name in EMBEDDER_REGISTRY and EMBEDDER_REGISTRY[name] != embedder_class:
        raise ValueError(f"Embedder '{name}' already registered with different class")

    # Basic interface check
    if not hasattr(embedder_class, "forward"):
        raise TypeError("Embedder class must implement 'forward' method")

    EMBEDDER_REGISTRY[name] = embedder_class
    return embedder_class


def register_transform(name: str, transform_class: type) -> type:
    """Register a video transform class.

    Args:
        name: Name to register under
        transform_class: Class implementing VideoTransform interface

    Returns:
        The registered class

    Example:
        >>> @register_transform("my_transform")
        ... class MyTransform(VideoTransform):
        ...     pass
    """
    if name in TRANSFORM_REGISTRY and TRANSFORM_REGISTRY[name] != transform_class:
        raise ValueError(f"Transform '{name}' already registered with different class")

    if not callable(transform_class):
        raise TypeError("Transform class must be callable")

    TRANSFORM_REGISTRY[name] = transform_class
    return transform_class


def register_loader(name: str, loader_class: type) -> type:
    """Register a video loader class.

    Args:
        name: Name to register under
        loader_class: Class implementing video loading interface

    Returns:
        The registered class
    """
    if name in LOADER_REGISTRY and LOADER_REGISTRY[name] != loader_class:
        raise ValueError(f"Loader '{name}' already registered with different class")

    LOADER_REGISTRY[name] = loader_class
    return loader_class


def register_metric(name: str, metric_class: type) -> type:
    """Register a video metric class.

    Args:
        name: Name to register under
        metric_class: Class implementing metric interface

    Returns:
        The registered class
    """
    if name in METRIC_REGISTRY and METRIC_REGISTRY[name] != metric_class:
        raise ValueError(f"Metric '{name}' already registered with different class")

    METRIC_REGISTRY[name] = metric_class
    return metric_class


def get_registered_embedders() -> dict[str, type]:
    """Get all registered embedder classes.

    Returns:
        Dictionary mapping names to embedder classes
    """
    return EMBEDDER_REGISTRY.copy()


def get_registered_transforms() -> dict[str, type]:
    """Get all registered transform classes.

    Returns:
        Dictionary mapping names to transform classes
    """
    return TRANSFORM_REGISTRY.copy()


def get_registered_loaders() -> dict[str, type]:
    """Get all registered loader classes.

    Returns:
        Dictionary mapping names to loader classes
    """
    return LOADER_REGISTRY.copy()


def get_registered_metrics() -> dict[str, type]:
    """Get all registered metric classes.

    Returns:
        Dictionary mapping names to metric classes
    """
    return METRIC_REGISTRY.copy()


def clear_registry(registry_name: str) -> None:
    """Clear a specific registry.

    Args:
        registry_name: Name of registry to clear

    Raises:
        ValueError: If registry_name not recognized

    Example:
        >>> clear_registry("embedders")  # Clear all embedders
    """
    registries = {
        "embedders": EMBEDDER_REGISTRY,
        "transforms": TRANSFORM_REGISTRY,
        "loaders": LOADER_REGISTRY,
        "metrics": METRIC_REGISTRY,
    }

    if registry_name not in registries:
        available = list(registries.keys())
        raise ValueError(f"Unknown registry '{registry_name}'. Available: {available}")

    registries[registry_name].clear()


def list_registered(registry_name: str) -> list[str]:
    """List names in a specific registry.

    Args:
        registry_name: Name of registry to list

    Returns:
        List of registered names

    Raises:
        ValueError: If registry_name not recognized
    """
    registries = {
        "embedders": EMBEDDER_REGISTRY,
        "transforms": TRANSFORM_REGISTRY,
        "loaders": LOADER_REGISTRY,
        "metrics": METRIC_REGISTRY,
    }

    if registry_name not in registries:
        available = list(registries.keys())
        raise ValueError(f"Unknown registry '{registry_name}'. Available: {available}")

    return list(registries[registry_name].keys())

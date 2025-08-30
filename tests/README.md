# Energy Transformer Test Suite

Comprehensive test suite for the Energy Transformer implementation.

## Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_attention.py    # Attention module tests
│   ├── test_hopfield.py     # Hopfield module tests
│   └── test_transformer.py  # Transformer block tests
├── integration/             # Integration and end-to-end tests
│   ├── test_equivalence.py  # Reference implementation equivalence
│   └── test_training.py     # Training workflow tests
├── conftest.py             # Shared pytest fixtures
└── test_all.py             # Main test runner
```

## Running Tests

Run all tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ -v --cov=associative --cov-report=term-missing
```

Run specific test modules:
```bash
pytest tests/unit/test_attention.py -v
pytest tests/integration/test_equivalence.py -v
```

Run tests matching a pattern:
```bash
pytest tests/ -k "energy" -v
pytest tests/ -k "TestGraphEnergyAttention" -v
```

## Test Categories

### Unit Tests
- Component initialization and configuration
- Forward pass shapes and types
- Gradient flow verification
- Numerical stability
- Parameter options (bias, dimensions)

### Integration Tests
- Equivalence with reference implementations
- Training workflows (MAE, classification)
- Energy minimization behavior
- Batch processing and masking

## Key Testing Principles

1. **Reproducibility**: Fixed seeds for deterministic behavior
2. **Isolation**: Each test is independent
3. **Coverage**: Both happy path and edge cases
4. **Performance**: Tests run quickly for rapid iteration

## Adding New Tests

1. Use pytest fixtures for common setups
2. Follow existing naming conventions
3. Document test purpose clearly
4. Verify both forward and backward passes
5. Check numerical stability
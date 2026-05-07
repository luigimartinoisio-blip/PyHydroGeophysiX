# PyHydroGeophysiX_Carl

Coupled hydrogeophysical modeling framework for the Carlone field site.

## Overview

PyHydroGeophysiX_Carl integrates field sensor data (ERT, water content, meteorological, cosmic-ray neutron sensors) with hydrological and geophysical simulations for coupled 2D hydrogeophysical modeling.

## Project Structure

```
├── src/                    # Core library
│   └── data_ingestion.py   # Data ingestion module
├── tests/                  # Test suite
├── examples/               # Example notebooks and scripts
├── data/                   # Production data (raw/processed)
├── models/                 # Model configurations
├── outputs/                # Generated outputs (excluded from git)
├── docs/                   # Documentation
└── pyproject.toml         # Project configuration
```

## Quick Start

1. Install dependencies:
   ```bash
   pip install -e .
   ```

2. Run data ingestion:
   ```bash
   python examples/scripts/build_database.py
   ```

3. Explore examples in `examples/notebooks/`

## Development

- Run tests: `pytest`
- Build docs: See `docs/`
- Code style: Black + Flake8

## License

[Add license information]
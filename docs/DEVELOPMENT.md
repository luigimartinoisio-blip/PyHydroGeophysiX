# Development Guide

## Setup

1. Clone the repository
2. Create virtual environment: `python -m venv .venv`
3. Activate: `.venv\Scripts\activate` (Windows)
4. Install: `pip install -e .[dev]`

## Testing

- Run all tests: `pytest`
- Run specific test: `pytest tests/unit/test_imports.py`
- Coverage: `pytest --cov=src`

## Code Style

- Format: `black src/ tests/ examples/`
- Lint: `flake8 src/ tests/ examples/`
- Type check: `mypy src/`

## Adding New Connectors

1. Extend `BaseConnector` in `src/data_ingestion.py`
2. Register with `ConnectorFactory`
3. Add unit tests in `tests/unit/`
4. Update documentation

## Contributing

1. Create feature branch
2. Write tests
3. Update docs
4. Submit PR
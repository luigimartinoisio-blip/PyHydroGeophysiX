# Testing Guide

## Test Structure

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # End-to-end pipeline tests
└── fixtures/       # Test data and fixtures
```

## Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/unit/test_imports.py

# With coverage
pytest --cov=src --cov-report=html

# Verbose output
pytest -v
```

## Writing Tests

- Unit tests: Test individual functions/classes
- Integration tests: Test full workflows
- Use fixtures for test data
- Mock external dependencies when possible

## Test Data

- Small datasets in `tests/fixtures/sample_data/`
- Use relative paths in tests
- Keep test data minimal but representative
import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture
def sample_data_path():
    """Path to sample test data"""
    return os.path.join(os.path.dirname(__file__), 'fixtures', 'sample_data')
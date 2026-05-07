import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
import pygimli as pg
import flopy
from data_ingestion import ConnectorFactory


def test_numpy_import():
    """Test NumPy import and version"""
    assert np.__version__ is not None
    print(f"✓ NumPy version: {np.__version__}")


def test_pygimli_import():
    """Test PyGIMLi import and version"""
    assert pg.__version__ is not None
    print(f"✓ PyGIMLi version: {pg.__version__}")


def test_flopy_import():
    """Test Flopy import and version"""
    assert flopy.__version__ is not None
    print(f"✓ Flopy version: {flopy.__version__}")


def test_connector_factory():
    """Test ConnectorFactory availability"""
    connectors = ConnectorFactory.available_connectors()
    assert isinstance(connectors, list)
    assert len(connectors) > 0
    print(f"✓ ConnectorFactory keys: {connectors}")


def test_pygimli_mesh_creation():
    """Test PyGIMLi mesh creation"""
    mesh = pg.createGrid(x=np.linspace(0, 10, 11), y=np.linspace(0, 10, 11))
    assert mesh.nodeCount() > 0
    print(f"✓ Mesh creata con successo: {mesh.nodeCount()} nodi.")


if __name__ == "__main__":
    print("Running import tests...")
    test_numpy_import()
    test_pygimli_import()
    test_flopy_import()
    test_connector_factory()
    test_pygimli_mesh_creation()
    print("All tests passed!")
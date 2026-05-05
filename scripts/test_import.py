import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pygimli as pg
import flopy
from data_ingestion import ConnectorFactory

print(f"NumPy version: {np.__version__}")
print(f"PyGIMLi version: {pg.__version__}")
print(f"Flopy version: {flopy.__version__}")
print(f"ConnectorFactory keys: {ConnectorFactory.available_connectors()}")

mesh = pg.createGrid(x=np.linspace(0, 10, 11), y=np.linspace(0, 10, 11))
print(f"Mesh creata con successo: {mesh.nodeCount()} nodi.")

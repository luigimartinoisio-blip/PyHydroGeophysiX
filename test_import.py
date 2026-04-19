import numpy as np
import pygimli as pg
import flopy
import PyHydroGeophysX as phgx

print(f"NumPy version: {np.__version__}")
print(f"PyGIMLi version: {pg.__version__}")
print(f"Flopy version: {flopy.__version__}")
print(f"PyHydroGeophysX version: {phgx.__version__}")

# Forza PyGIMLi a eseguire un calcolo base per testare i binari C++
mesh = pg.createGrid(x=np.linspace(0, 10, 11), y=np.linspace(0, 10, 11))
print(f"Mesh creata con successo: {mesh.nodeCount()} nodi.")

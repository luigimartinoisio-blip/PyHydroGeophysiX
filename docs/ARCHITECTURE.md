# Architecture Overview

## System Design

PyHydroGeophysiX_Carl follows a layered architecture for coupled hydrogeophysical modeling:

### Layer 1: Data Ingestion
- **Location**: `src/data_ingestion.py`
- **Purpose**: Standardized ingestion from heterogeneous field sensors
- **Connectors**: ERT, COSMOS, Water Level, Meteorological, Spatial
- **Output**: Parquet files in `data/processed/`

### Layer 2: Coupled Modeling
- **Location**: `examples/notebooks/`
- **Purpose**: Hydrogeophysical inversion and simulation
- **Tools**: PyGIMLi (geophysics) + Flopy (hydrogeology)
- **Output**: Model results in `outputs/`

## Data Flow

```
Raw Field Data → Data Ingestion → Processed Parquet → Coupled Modeling → Results
```

## Key Components

- **ConnectorFactory**: Registry pattern for sensor-specific connectors
- **BaseConnector**: Abstract interface for all data sources
- **MODFLOW Integration**: Groundwater flow simulation
- **ERT Inversion**: Electrical resistivity tomography
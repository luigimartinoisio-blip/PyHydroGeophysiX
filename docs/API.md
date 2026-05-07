# API Reference

## Data Ingestion Module

### ConnectorFactory

Registry for data connectors.

```python
from src.data_ingestion import ConnectorFactory

# List available connectors
connectors = ConnectorFactory.available_connectors()

# Get connector instance
connector = ConnectorFactory.create('ert', config)
```

### BaseConnector

Abstract base class for all data connectors.

**Methods:**
- `load_data(path: Path) -> pd.DataFrame`: Load data from file
- `validate_data(df: pd.DataFrame) -> bool`: Validate data format
- `process_data(df: pd.DataFrame) -> pd.DataFrame`: Process/clean data

### Concrete Connectors

- **ERTConnector**: Electrical Resistivity Tomography data
- **CosmosConnector**: Cosmic-ray neutron sensor data
- **MeterConnector**: Water level/pressure data
- **MeteoConnector**: Meteorological data
- **SpatialConnector**: Electrode positions and topography

## Usage Examples

See `examples/notebooks/` for complete usage examples.
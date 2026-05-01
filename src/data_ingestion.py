"""
PyHydroGeophysiX – Data Ingestion Engine
=========================================
Macro-Area 1 · Layer 1: Modular, asynchronous-ready time-series ingestion
from heterogeneous field sensors.

Architecture
------------
* BaseConnector       – Abstract base class defining the ingestion contract.
* MeterConnector      – Water-level / pressure transducer data (CSV).
* ERTConnector        – Electrical Resistivity Tomography survey exports (CSV).
* MeteoConnector      – Meteorological station data (CSV).
* CosmosConnector     – Cosmic-Ray Neutron Sensor data (CSV).
* ConnectorFactory    – Instantiates the correct connector from a string key.

Constraints
-----------
* Python 3.10+  |  full type hints  |  pathlib  |  pandas DatetimeIndex
* No resampling or spatial alignment performed here.
* Missing timezone info → UTC assumed.

Usage
-----
    from src.data_ingestion import ConnectorFactory
    from pathlib import Path

    connector = ConnectorFactory.get("meter")
    df = connector.parse_data(Path("data/raw/meter_001.csv"))
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# Only add a handler when the module is the entry point (not when imported).
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logger.addHandler(_handler)

logger.setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Abstract Base Class
# ---------------------------------------------------------------------------


class BaseConnector(ABC):
    """Contract for every field-sensor connector.

    Subclasses **must** implement :meth:`parse_data`, which reads a raw data
    file and returns a timezone-aware :class:`pandas.DataFrame` with a
    :class:`pandas.DatetimeIndex`.
    """

    #: Human-readable sensor type label – override in each subclass.
    sensor_type: str = "unknown"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} sensor_type={self.sensor_type!r}>"

    @abstractmethod
    def parse_data(self, file_path: Path) -> pd.DataFrame:
        """Parse *file_path* and return a standardised DataFrame.

        Parameters
        ----------
        file_path:
            Absolute or relative path to the raw data file.

        Returns
        -------
        pandas.DataFrame
            - Index : :class:`pandas.DatetimeIndex`, timezone-aware (UTC).
            - Columns : sensor-specific measurements (no fixed schema here).

        Raises
        ------
        FileNotFoundError
            If *file_path* does not exist.
        ValueError
            If the file cannot be parsed as expected.
        """

    # ------------------------------------------------------------------
    # Shared helpers available to every subclass
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve(file_path: Path) -> Path:
        """Return an absolute Path, raising :exc:`FileNotFoundError` early."""
        p = Path(file_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {p}")
        return p

    @staticmethod
    def _ensure_utc(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Localise a tz-naive index to UTC; keep a tz-aware one as-is."""
        if index.tz is None:
            logger.debug("DatetimeIndex has no timezone – assuming UTC.")
            return index.tz_localize("UTC")
        return index.tz_convert("UTC")


# ---------------------------------------------------------------------------
# Concrete Connectors
# ---------------------------------------------------------------------------


class MeterConnector(BaseConnector):
    """Ingests water-level / pressure-transducer CSV exports.

    Expected CSV layout (header on row 0):

    .. code-block:: text

        timestamp,water_level_m,temperature_c
        2024-01-01T00:00:00,1.234,12.5
        ...

    The ``timestamp`` column is parsed as the index.  Any additional columns
    are returned as-is.
    """

    sensor_type = "meter"

    #: Name of the column to be used as the datetime index.
    _TIMESTAMP_COL: str = "timestamp"

    def parse_data(self, file_path: Path) -> pd.DataFrame:
        """Parse a meter CSV file.

        Parameters
        ----------
        file_path:
            Path to the ``.csv`` data file.

        Returns
        -------
        pandas.DataFrame
            Index : DatetimeIndex (UTC).
            Columns : all measurement columns present in the file.
        """
        p = self._resolve(file_path)
        logger.info("[MeterConnector] Ingesting: %s", p)

        try:
            df = pd.read_csv(
                p,
                parse_dates=[self._TIMESTAMP_COL],
                index_col=self._TIMESTAMP_COL,
            )
        except KeyError as exc:
            raise ValueError(
                f"[MeterConnector] Expected column '{self._TIMESTAMP_COL}' "
                f"not found in {p}."
            ) from exc
        except Exception as exc:
            raise ValueError(
                f"[MeterConnector] Failed to parse {p}: {exc}"
            ) from exc

        df.index = self._ensure_utc(df.index)
        df.index.name = "datetime_utc"
        logger.info(
            "[MeterConnector] Parsed %d rows, columns: %s",
            len(df),
            df.columns.tolist(),
        )
        return df


class ERTConnector(BaseConnector):
    """Ingests Electrical Resistivity Tomography survey CSV exports.

    Expected CSV layout:

    .. code-block:: text

        datetime,electrode_a,electrode_b,electrode_m,electrode_n,resistance_ohm,apparent_resistivity_ohmm
        2024-06-15T08:00:00,1,2,3,4,12.3,45.6
        ...
    """

    sensor_type = "ert"

    _TIMESTAMP_COL: str = "datetime"

    def parse_data(self, file_path: Path) -> pd.DataFrame:
        """Parse an ERT CSV file.

        Returns
        -------
        pandas.DataFrame
            Index : DatetimeIndex (UTC).
            Columns : electrode configuration + measured resistivity values.
        """
        p = self._resolve(file_path)
        logger.info("[ERTConnector] Ingesting: %s", p)

        try:
            df = pd.read_csv(
                p,
                parse_dates=[self._TIMESTAMP_COL],
                index_col=self._TIMESTAMP_COL,
            )
        except KeyError as exc:
            raise ValueError(
                f"[ERTConnector] Expected column '{self._TIMESTAMP_COL}' "
                f"not found in {p}."
            ) from exc
        except Exception as exc:
            raise ValueError(
                f"[ERTConnector] Failed to parse {p}: {exc}"
            ) from exc

        df.index = self._ensure_utc(df.index)
        df.index.name = "datetime_utc"
        logger.info(
            "[ERTConnector] Parsed %d rows, columns: %s",
            len(df),
            df.columns.tolist(),
        )
        return df


class MeteoConnector(BaseConnector):
    """Ingests meteorological station CSV exports.

    Expected CSV layout:

    .. code-block:: text

        obs_time,air_temp_c,rel_humidity_pct,precip_mm,wind_speed_ms,solar_rad_wm2
        2024-01-01T00:00:00,5.2,88.0,0.0,2.1,0.0
        ...
    """

    sensor_type = "meteo"

    _TIMESTAMP_COL: str = "obs_time"

    def parse_data(self, file_path: Path) -> pd.DataFrame:
        """Parse a meteorological station CSV file.

        Returns
        -------
        pandas.DataFrame
            Index : DatetimeIndex (UTC).
            Columns : atmospheric variables present in the file.
        """
        p = self._resolve(file_path)
        logger.info("[MeteoConnector] Ingesting: %s", p)

        try:
            df = pd.read_csv(
                p,
                parse_dates=[self._TIMESTAMP_COL],
                index_col=self._TIMESTAMP_COL,
            )
        except KeyError as exc:
            raise ValueError(
                f"[MeteoConnector] Expected column '{self._TIMESTAMP_COL}' "
                f"not found in {p}."
            ) from exc
        except Exception as exc:
            raise ValueError(
                f"[MeteoConnector] Failed to parse {p}: {exc}"
            ) from exc

        df.index = self._ensure_utc(df.index)
        df.index.name = "datetime_utc"
        logger.info(
            "[MeteoConnector] Parsed %d rows, columns: %s",
            len(df),
            df.columns.tolist(),
        )
        return df


class CosmosConnector(BaseConnector):
    """Ingests Cosmic-Ray Neutron Sensor (CRNS) CSV exports.

    Expected CSV layout:

    .. code-block:: text

        utc_time,raw_neutron_count,corrected_neutron_count,soil_moisture_vol
        2024-01-01T01:00:00,1500,1480,0.32
        ...
    """

    sensor_type = "cosmos"

    _TIMESTAMP_COL: str = "utc_time"

    def parse_data(self, file_path: Path) -> pd.DataFrame:
        """Parse a CRNS CSV file.

        Returns
        -------
        pandas.DataFrame
            Index : DatetimeIndex (UTC).
            Columns : neutron count + derived soil moisture.
        """
        p = self._resolve(file_path)
        logger.info("[CosmosConnector] Ingesting: %s", p)

        try:
            df = pd.read_csv(
                p,
                parse_dates=[self._TIMESTAMP_COL],
                index_col=self._TIMESTAMP_COL,
            )
        except KeyError as exc:
            raise ValueError(
                f"[CosmosConnector] Expected column '{self._TIMESTAMP_COL}' "
                f"not found in {p}."
            ) from exc
        except Exception as exc:
            raise ValueError(
                f"[CosmosConnector] Failed to parse {p}: {exc}"
            ) from exc

        df.index = self._ensure_utc(df.index)
        df.index.name = "datetime_utc"
        logger.info(
            "[CosmosConnector] Parsed %d rows, columns: %s",
            len(df),
            df.columns.tolist(),
        )
        return df


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

#: Registry mapping string keys → connector classes.
_CONNECTOR_REGISTRY: dict[str, type[BaseConnector]] = {
    "meter": MeterConnector,
    "ert": ERTConnector,
    "meteo": MeteoConnector,
    "cosmos": CosmosConnector,
}


class ConnectorFactory:
    """Instantiate the correct :class:`BaseConnector` subclass by key.

    Example
    -------
    >>> connector = ConnectorFactory.get("meter")
    >>> type(connector)
    <class 'src.data_ingestion.MeterConnector'>
    """

    @staticmethod
    def get(sensor_type: str) -> BaseConnector:
        """Return a connector instance for *sensor_type*.

        Parameters
        ----------
        sensor_type:
            Case-insensitive sensor identifier. Supported values:
            ``'meter'``, ``'ert'``, ``'meteo'``, ``'cosmos'``.

        Returns
        -------
        BaseConnector
            A fresh connector instance ready to call :meth:`~BaseConnector.parse_data`.

        Raises
        ------
        KeyError
            If *sensor_type* is not registered.
        """
        key = sensor_type.strip().lower()
        if key not in _CONNECTOR_REGISTRY:
            available = ", ".join(sorted(_CONNECTOR_REGISTRY))
            raise KeyError(
                f"Unknown sensor type '{sensor_type}'. "
                f"Available connectors: [{available}]."
            )
        cls = _CONNECTOR_REGISTRY[key]
        logger.debug("[ConnectorFactory] Instantiating connector: %s", cls.__name__)
        return cls()

    @staticmethod
    def available_connectors() -> list[str]:
        """Return the list of registered sensor type keys."""
        return sorted(_CONNECTOR_REGISTRY.keys())

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
* CosmosConnector     – Cosmic-Ray Neutron Sensor data (CSV, Italian dates).
* SpatialConnector    – Electrode / topography position files (CSV, #,X,Y,Z,Depth).
* SequenceConnector   – ERT survey sequence files (tab-delimited, two-block format).
* ConnectorFactory    – Instantiates the correct connector from a string key.

Constraints
-----------
* Python 3.10+  |  full type hints  |  pathlib  |  pandas DatetimeIndex / RangeIndex
* No resampling or spatial alignment performed here.
* Missing timezone info → UTC assumed.

Usage
-----
    from src.data_ingestion import ConnectorFactory
    from pathlib import Path

    # Time-series connectors
    connector = ConnectorFactory.get("cosmos")
    df = connector.parse_data(Path("data/raw/COSMOS_SOIL_MOISTURE/COSMOS-Carlone.xlsx.csv"))

    # Spatial connector (topography / electrode positions)
    connector = ConnectorFactory.get("spatial")
    df = connector.parse_data(Path("data/raw/Position/08topog_TL.txt"))

    # Sequence / metadata connector
    connector = ConnectorFactory.get("sequence")
    elec_df, seq_df = connector.parse_data(
        Path("data/raw/sequence/2DDsup+MG+DDsup_rec.txt")
    )
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import NamedTuple

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
# Italian month abbreviation map
# ---------------------------------------------------------------------------
#: Maps Italian three-letter month abbreviations to their English equivalents.
#: Used by CosmosConnector to normalise dates like "10-lug-25" → "10-Jul-25".
_ITALIAN_MONTHS: dict[str, str] = {
    "gen": "Jan",
    "feb": "Feb",
    "mar": "Mar",
    "apr": "Apr",
    "mag": "May",
    "giu": "Jun",
    "lug": "Jul",
    "ago": "Aug",
    "set": "Sep",
    "ott": "Oct",
    "nov": "Nov",
    "dic": "Dec",
}

# Pre-compiled regex for Italian date substitution: matches any 3-letter
# lowercase sequence surrounded by hyphens (the month component).
_IT_MONTH_RE = re.compile(
    r"-(" + "|".join(_ITALIAN_MONTHS.keys()) + r")-",
    flags=re.IGNORECASE,
)


def _translate_italian_dates(series: pd.Series) -> pd.Series:
    """Replace Italian month abbreviations in a string Series with English ones.

    Example: ``'10-lug-25'`` → ``'10-Jul-25'``

    Parameters
    ----------
    series:
        A pandas Series of raw date strings (e.g., the ``Date`` column from a
        COSMOS export).

    Returns
    -------
    pandas.Series
        The same Series with Italian month names translated to English.
    """

    def _replace(match: re.Match) -> str:  # type: ignore[type-arg]
        it_abbr = match.group(1).lower()
        return f"-{_ITALIAN_MONTHS[it_abbr]}-"

    return series.apply(lambda s: _IT_MONTH_RE.sub(_replace, str(s)))


# ---------------------------------------------------------------------------
# Abstract Base Class
# ---------------------------------------------------------------------------


class BaseConnector(ABC):
    """Contract for every field-sensor connector.

    Subclasses **must** implement :meth:`parse_data`, which reads a raw data
    file and returns a standardised :class:`pandas.DataFrame`.

    For time-series connectors the returned DataFrame must have a
    timezone-aware :class:`pandas.DatetimeIndex`.  Spatial and sequence
    connectors use integer indices instead (see :class:`SpatialConnector` and
    :class:`SequenceConnector`).
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
            Shape and index type depend on the subclass.  See each connector's
            docstring for the specific contract.

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
        except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
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
        except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
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
        except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
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
    """Ingests Cosmic-Ray Neutron Sensor (CRNS) CSV exports with Italian dates.

    Expected CSV layout (COSMOS-Carlone format):

    .. code-block:: text

        Date,COSMOS Daily mean Volumetric SM,Cumulated precipitation [mm]
        10-lug-25,18.54957647,0
        11-lug-25,17.73872711,0
        ...

    Date format
    -----------
    The ``Date`` column uses Italian three-letter month abbreviations in the
    pattern ``DD-mon-YY``.  The complete mapping is::

        gen → Jan  feb → Feb  mar → Mar  apr → Apr
        mag → May  giu → Jun  lug → Jul  ago → Aug
        set → Sep  ott → Oct  nov → Nov  dic → Dec

    The parser translates these abbreviations to English before handing the
    column to :func:`pandas.to_datetime` with ``dayfirst=True``.

    Returns
    -------
    pandas.DataFrame
        Index : DatetimeIndex (UTC, daily frequency).
        Columns:

        * ``cosmos_vsm``   – Daily mean volumetric soil moisture (%).
        * ``precip_cum_mm`` – Cumulated precipitation (mm).
    """

    sensor_type = "cosmos"

    #: Raw header name for the date column.
    _DATE_COL: str = "Date"
    #: Expected raw column names (in order).
    _RAW_COLS: tuple[str, str, str] = (
        "Date",
        "COSMOS Daily mean Volumetric SM",
        "Cumulated precipitation [mm]",
    )
    #: Canonical output column names mapped from the raw ones.
    _COL_RENAME: dict[str, str] = {
        "COSMOS Daily mean Volumetric SM": "cosmos_vsm",
        "Cumulated precipitation [mm]": "precip_cum_mm",
    }

    def parse_data(self, file_path: Path) -> pd.DataFrame:
        """Parse an Italian-date COSMOS CSV file.

        Parameters
        ----------
        file_path:
            Path to the ``.csv`` data file.

        Returns
        -------
        pandas.DataFrame
            Index : DatetimeIndex (UTC, daily).
            Columns : ``cosmos_vsm``, ``precip_cum_mm``.

        Raises
        ------
        FileNotFoundError
            If *file_path* does not exist.
        ValueError
            If the expected columns are not found or the dates cannot be parsed.
        """
        p = self._resolve(file_path)
        logger.info("[CosmosConnector] Ingesting: %s", p)

        # ── 1. Read raw CSV without any date parsing ─────────────────────────
        try:
            df_raw = pd.read_csv(p, dtype=str)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
            raise ValueError(
                f"[CosmosConnector] Failed to read CSV {p}: {exc}"
            ) from exc

        # ── 2. Validate expected columns ─────────────────────────────────────
        missing = [c for c in self._RAW_COLS if c not in df_raw.columns]
        if missing:
            raise ValueError(
                f"[CosmosConnector] Missing expected columns in {p}: {missing}. "
                f"Found: {df_raw.columns.tolist()}"
            )

        # ── 3. Translate Italian month abbreviations → English ───────────────
        df_raw[self._DATE_COL] = _translate_italian_dates(df_raw[self._DATE_COL])
        logger.debug(
            "[CosmosConnector] Sample dates after translation: %s",
            df_raw[self._DATE_COL].head(3).tolist(),
        )

        # ── 4. Parse dates (DD-Mon-YY, dayfirst=True) ───────────────────────
        try:
            dates = pd.to_datetime(
                df_raw[self._DATE_COL],
                dayfirst=True,
                format="%d-%b-%y",
                errors="coerce",
            )
        except Exception as exc:
            raise ValueError(
                f"[CosmosConnector] Date parsing failed in {p}: {exc}"
            ) from exc

        nat_count = int(dates.isna().sum())
        if nat_count:
            logger.warning(
                "[CosmosConnector] %d date(s) could not be parsed and will be dropped.",
                nat_count,
            )

        # ── 5. Build clean DataFrame ─────────────────────────────────────────
        df = df_raw.drop(columns=[self._DATE_COL]).copy()
        df = df.rename(columns=self._COL_RENAME)

        # Convert measurement columns to float
        for col in self._COL_RENAME.values():
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.index = pd.DatetimeIndex(dates)
        df.index = self._ensure_utc(df.index)
        df.index.name = "datetime_utc"

        # Drop rows with unparseable dates (NaT) or all-NaN measurements
        df = df[df.index.notna()]
        df = df.dropna(how="all")

        logger.info(
            "[CosmosConnector] Parsed %d rows, columns: %s",
            len(df),
            df.columns.tolist(),
        )
        return df


class SpatialConnector(BaseConnector):
    """Ingests electrode / topography position files.

    Handles two sub-formats automatically detected from the file header:

    1. **Topography / electrode grid** (``08topog_TL.txt`` style):

       .. code-block:: text

           #,X,Y,Z,Depth
           1,0.00,0.00,686.17,0.00
           2,1.20,0.00,686.00,0.00
           ...

       Comma-separated; header contains ``#,X,Y,Z``.  Index = electrode number
       (``#``).  The optional ``Depth`` column is kept if present.

    2. **GPS / geodetic coordinates** (``electr_coord.csv`` style):

       .. code-block:: text

           Nome punto,Latitudine,Longitudine,Altezza ellisoidica
           Pt1,40.342307,15.403299,686.175
           ...

       The ``Nome punto`` column is stripped of the leading ``Pt`` prefix,
       cast to integer, and used as the index.

    Returns
    -------
    pandas.DataFrame
        Index name : ``electrode_id`` (integer, 1-based).
        Columns: depend on sub-format (``X``, ``Y``, ``Z``, ``Depth``
        **or** ``Latitudine``, ``Longitudine``, ``Altezza ellisoidica``).
    """

    sensor_type = "spatial"

    #: Column expected in topography-style files (comma-separated, no spaces).
    _TOPO_INDEX_COL: str = "#"
    #: Column expected in GPS-coordinate files.
    _GPS_NAME_COL: str = "Nome punto"

    def parse_data(self, file_path: Path) -> pd.DataFrame:
        """Parse a topography or electrode-coordinate file.

        Parameters
        ----------
        file_path:
            Path to the position file (either ``.txt`` or ``.csv``).

        Returns
        -------
        pandas.DataFrame
            Index : integer electrode number named ``'electrode_id'``.
            Columns : spatial coordinates.

        Raises
        ------
        FileNotFoundError
            If *file_path* does not exist.
        ValueError
            If the file format is not recognised.
        """
        p = self._resolve(file_path)
        logger.info("[SpatialConnector] Ingesting: %s", p)

        # ── Peek at the header to decide the sub-format ──────────────────────
        with open(p, encoding="utf-8", errors="replace") as fh:
            header_line = fh.readline().rstrip("\r\n")

        if self._TOPO_INDEX_COL in header_line.split(","):
            df = self._parse_topography(p)
        elif self._GPS_NAME_COL in header_line:
            df = self._parse_gps(p)
        else:
            raise ValueError(
                f"[SpatialConnector] Unrecognised header format in {p}: "
                f"'{header_line}'"
            )

        logger.info(
            "[SpatialConnector] Parsed %d electrodes, columns: %s",
            len(df),
            df.columns.tolist(),
        )
        return df

    # ------------------------------------------------------------------
    # Private sub-parsers
    # ------------------------------------------------------------------

    def _parse_topography(self, p: Path) -> pd.DataFrame:
        """Parse a ``#,X,Y,Z[,Depth]`` comma-separated file.

        Lines that begin with ``#`` (including the header) are treated as
        comments and skipped; column names are derived from the first such
        header line so the optional ``Depth`` column is handled automatically.
        """
        with open(p, encoding="utf-8", errors="replace") as fh:
            raw_header = fh.readline().rstrip("\r\n")
        col_names = [c.strip() for c in raw_header.lstrip("#").split(",") if c.strip()]
        df = pd.read_csv(
            p,
            comment="#",
            header=None,
            names=[self._TOPO_INDEX_COL] + col_names,
            sep=",",
            skipinitialspace=True,
        )
        df = df.set_index(self._TOPO_INDEX_COL)
        df.index = df.index.astype(int)
        df.index.name = "electrode_id"
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(how="all")
        return df

    def _parse_gps(self, p: Path) -> pd.DataFrame:
        """Parse a ``Nome punto, Latitudine, Longitudine, Altezza`` CSV file."""
        df = pd.read_csv(p)
        # Extract integer electrode index from "PtN" labels
        df.index = (
            df[self._GPS_NAME_COL]
            .str.replace(r"(?i)^pt", "", regex=True)
            .astype(int)
        )
        df.index.name = "electrode_id"
        df = df.drop(columns=[self._GPS_NAME_COL])
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(how="all")
        return df


class SequenceData(NamedTuple):
    """Return value of :meth:`SequenceConnector.parse_data`.

    Supports tuple unpacking: ``electrodes_df, sequence_df = connector.parse_data(path)``
    """

    electrodes: pd.DataFrame
    sequence: pd.DataFrame


class SequenceConnector(BaseConnector):
    """Ingests ERT survey sequence / electrode-grid files.

    The target file (e.g., ``2DDsup+MG+DDsup_rec.txt``) uses a **two-block**
    tab-delimited format:

    .. code-block:: text

        #   X       Y       Z
        1   0.00    0.00    0.00
        2   1.20    0.00    0.00
        ...
        48  56.40   0.00    0.00
        #   A   B   M   N
        1   1   2   3   4
        2   1   2   4   5
        ...

    Each block begins with a ``#``-prefixed header line.  The first block
    defines the electrode grid (columns ``X``, ``Y``, ``Z``); the second block
    defines the measurement sequence (columns ``A``, ``B``, ``M``, ``N``).

    .. note::
        This connector deviates from :class:`BaseConnector`'s standard contract
        in that :meth:`parse_data` returns a **tuple** of two DataFrames
        ``(electrodes_df, sequence_df)`` rather than a single DataFrame.
        The factory key ``'sequence'`` documents this deviation.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        * **electrodes_df** – Electrode grid.  Index = electrode number
          (``electrode_id``).  Columns: ``X``, ``Y``, ``Z``.
        * **sequence_df** – Measurement sequence.  Index = measurement number
          (``measurement_id``).  Columns: ``A``, ``B``, ``M``, ``N``
          (integer electrode indices).
    """

    sensor_type = "sequence"

    def parse_data(  # type: ignore[override]
        self, file_path: Path
    ) -> SequenceData:
        """Parse a two-block ERT sequence file.

        Parameters
        ----------
        file_path:
            Path to the ``.txt`` sequence file.

        Returns
        -------
        SequenceData
            Named tuple with fields ``electrodes`` and ``sequence``.
            Supports unpacking: ``electrodes_df, sequence_df = connector.parse_data(path)``.

        Raises
        ------
        FileNotFoundError
            If *file_path* does not exist.
        ValueError
            If fewer than two ``#``-header blocks are found in the file.
        """
        p = self._resolve(file_path)
        logger.info("[SequenceConnector] Ingesting: %s", p)

        # ── Read all lines ────────────────────────────────────────────────────
        with open(p, encoding="utf-8", errors="replace") as fh:
            raw_lines = fh.readlines()

        # ── Split into blocks separated by lines that start with '#' ─────────
        block_starts: list[int] = [
            i for i, ln in enumerate(raw_lines) if ln.startswith("#")
        ]

        if len(block_starts) < 2:
            raise ValueError(
                f"[SequenceConnector] Expected at least 2 header blocks in {p}, "
                f"found {len(block_starts)}."
            )

        def _parse_block(start: int, end: int | None, index_name: str) -> pd.DataFrame:
            """Parse one tab-delimited block into a DataFrame."""
            # The header line is e.g. "#\tX\tY\tZ".  lstrip('#') leaves "\tX\tY\tZ";
            # splitting on \t gives ['', 'X', 'Y', 'Z'] or ['X', 'Y', 'Z'] depending
            # on whitespace.  We prepend '#' explicitly so the column count matches
            # the data rows (which always carry the index as the first tab field).
            raw_header = raw_lines[start].lstrip("#").strip()
            header_fields = ["#"] + [h for h in raw_header.split("\t") if h]
            data_lines = raw_lines[start + 1 : end]
            rows = []
            skipped = 0
            for ln in data_lines:
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split("\t")
                if len(parts) != len(header_fields):
                    logger.warning(
                        "[SequenceConnector] Skipping malformed line: %r", ln
                    )
                    skipped += 1
                    continue
                rows.append(parts)
            if not rows:
                raise ValueError(
                    f"[SequenceConnector] Block starting at line {start + 1} "
                    f"in {p} produced no valid rows "
                    f"({skipped} malformed lines skipped)."
                )
            df = pd.DataFrame(rows, columns=header_fields)
            # The first column is always the sequential index '#'
            df[header_fields[0]] = pd.to_numeric(df[header_fields[0]], errors="coerce")
            df = df.set_index(header_fields[0])
            df.index = df.index.astype(int)
            df.index.name = index_name
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(how="all")
            return df

        # Block 1: electrode grid  (lines block_starts[0] … block_starts[1])
        electrodes_df = _parse_block(
            block_starts[0], block_starts[1], index_name="electrode_id"
        )
        # Block 2: measurement sequence  (lines block_starts[1] … EOF)
        sequence_df = _parse_block(
            block_starts[1], None, index_name="measurement_id"
        )
        # Cast electrode index columns to int – raise clearly if NaN remain after coerce
        for col in ["A", "B", "M", "N"]:
            if col in sequence_df.columns:
                nan_mask = sequence_df[col].isna()
                if nan_mask.any():
                    bad_idx = sequence_df.index[nan_mask].tolist()
                    raise ValueError(
                        f"[SequenceConnector] Column '{col}' contains non-numeric "
                        f"values at measurement_id(s) {bad_idx} in {p}."
                    )
                sequence_df[col] = sequence_df[col].astype(int)

        logger.info(
            "[SequenceConnector] Electrodes: %d, Measurements: %d",
            len(electrodes_df),
            len(sequence_df),
        )
        return SequenceData(electrodes=electrodes_df, sequence=sequence_df)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

#: Registry mapping string keys → connector classes.
_CONNECTOR_REGISTRY: dict[str, type[BaseConnector]] = {
    "meter": MeterConnector,
    "ert": ERTConnector,
    "meteo": MeteoConnector,
    "cosmos": CosmosConnector,
    "spatial": SpatialConnector,
    "sequence": SequenceConnector,
}


class ConnectorFactory:
    """Instantiate the correct :class:`BaseConnector` subclass by key.

    Example
    -------
    >>> connector = ConnectorFactory.get("cosmos")
    >>> type(connector)
    <class 'src.data_ingestion.CosmosConnector'>

    Supported keys
    --------------
    ``'meter'``, ``'ert'``, ``'meteo'``, ``'cosmos'``,
    ``'spatial'``, ``'sequence'``.
    """

    @staticmethod
    def get(sensor_type: str) -> BaseConnector:
        """Return a connector instance for *sensor_type*.

        Parameters
        ----------
        sensor_type:
            Case-insensitive sensor identifier.

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

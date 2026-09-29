"""
GeoInsight-RigX (eRTMAC-NWIS)
Core Engine - Well Log Parser & Data Cleaning Pipeline
"""

import os
import logging
from pathlib import Path
from typing import Union, Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import lasio
from scipy.signal import medfilt
from scipy.ndimage import median_filter

# Configure logger
logger = logging.getLogger("WellLogParser")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class WellLogParser:
    """
    Robust Well Log Parser and Preprocessing Engine.
    Handles ingestion of raw .LAS (v1.2, v2.0, v3.0) and .CSV well logs,
    null value handling, depth interval standardization to meters,
    and rolling median filtering via SciPy for sensor noise reduction.
    """

    FEET_TO_METERS = 0.3048
    COMMON_NULL_VALUES = [-999.25, -999.0, -9999.0, -99999.0, 999.25, 9999.0]
    DEPTH_ALIASES = ["dept", "depth", "md", "measured_depth", "tvd", "tvdss", "depth_m", "depth_ft"]

    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        """
        Initialize parser with default base directory.
        Defaults to project's data_source/well_logs.
        """
        if data_dir is not None:
            self.data_dir = Path(data_dir)
        else:
            # Default to relative data_source/well_logs or environment variable
            base_project_dir = Path(__file__).resolve().parent.parent
            default_well_logs = os.getenv("WELL_LOGS_DATA_DIR", str(base_project_dir / "data_source" / "well_logs"))
            self.data_dir = Path(default_well_logs)

        self.last_metadata: Dict[str, Any] = {}

    def _resolve_file_path(self, filepath: Union[str, Path]) -> Path:
        """Resolve file path either as absolute or relative to self.data_dir."""
        path = Path(filepath)
        if not path.is_file():
            candidate = self.data_dir / filepath
            if candidate.is_file():
                return candidate
            raise FileNotFoundError(f"Well log file not found at '{filepath}' or '{candidate}'")
        return path

    def _detect_depth_column(self, df: pd.DataFrame) -> Optional[str]:
        """Detect the depth column name regardless of casing and common naming conventions."""
        lower_cols = {col.lower(): col for col in df.columns}
        for alias in self.DEPTH_ALIASES:
            if alias in lower_cols:
                return lower_cols[alias]
        # Partial match if exact alias not found
        for col, orig in lower_cols.items():
            if "depth" in col or "dept" in col:
                return orig
        return None

    def ingest_las(self, filepath: Union[str, Path]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Ingest a .LAS file using lasio.
        Extracts curve data into a DataFrame and preserves header metadata.
        """
        resolved_path = self._resolve_file_path(filepath)
        logger.info(f"Ingesting LAS file: {resolved_path.name}")

        try:
            las = lasio.read(str(resolved_path))
        except Exception as e:
            logger.error(f"Failed to read LAS file {resolved_path}: {e}")
            raise

        # Convert to pandas DataFrame
        df = las.df().reset_index()

        # Extract well and curve metadata
        well_meta = {}
        for item in las.well:
            well_meta[item.mnemonic.upper()] = {
                "value": item.value,
                "unit": item.unit,
                "descr": item.descr,
            }

        curve_meta = {}
        for curve in las.curves:
            curve_meta[curve.mnemonic.upper()] = {
                "unit": curve.unit,
                "descr": curve.descr,
                "data_points": len(curve.data),
            }

        metadata = {
            "source_type": "LAS",
            "version": getattr(las.version, "VERS", {}).value if hasattr(las, "version") else "unknown",
            "well_name": well_meta.get("WELL", {}).get("value", resolved_path.stem),
            "uwi": well_meta.get("UWI", {}).get("value", "UNKNOWN"),
            "well_metadata": well_meta,
            "curve_metadata": curve_meta,
            "original_null": getattr(las.well, "NULL", None).value if "NULL" in well_meta else -999.25,
            "file_path": str(resolved_path),
        }

        # Store last metadata and ensure strict float64 type-casting
        df = self.enforce_type_casting(df)
        self.last_metadata = metadata
        return df, metadata

    def ingest_csv(
        self,
        filepath: Union[str, Path],
        depth_col: Optional[str] = None,
        null_values: Optional[List[float]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Ingest a CSV well log file.
        Detects depth column, converts numerical data, and packages basic metadata.
        """
        resolved_path = self._resolve_file_path(filepath)
        logger.info(f"Ingesting CSV log file: {resolved_path.name}")

        # Attempt reading CSV
        df = pd.read_csv(resolved_path)

        # Detect depth column if not explicitly given
        detected_depth = depth_col or self._detect_depth_column(df)
        if detected_depth is None:
            logger.warning(f"Could not automatically detect depth column in {resolved_path.name}")

        metadata = {
            "source_type": "CSV",
            "well_name": resolved_path.stem,
            "uwi": "UNKNOWN",
            "depth_col": detected_depth,
            "columns": list(df.columns),
            "file_path": str(resolved_path),
        }

        df = self.enforce_type_casting(df)
        self.last_metadata = metadata
        return df, metadata

    def standardize_depth(
        self,
        df: pd.DataFrame,
        current_unit: Optional[str] = None,
        depth_col: Optional[str] = None,
        target_col_name: str = "DEPTH_M",
    ) -> pd.DataFrame:
        """
        Standardize depth column to meters:
        - Auto-detects feet vs meters if current_unit not provided (via metadata or column naming).
        - Multiplies feet by 0.3048 to obtain meters.
        - Renames or designates the depth column to target_col_name (default: DEPTH_M).
        - Sorts DataFrame monotonically ascending by depth.
        """
        df_out = df.copy()
        actual_depth_col = depth_col or self._detect_depth_column(df_out)

        if actual_depth_col is None:
            raise ValueError("No depth column detected to standardize.")

        # Determine unit
        unit = current_unit
        if unit is None:
            # Check last metadata for curve unit
            if self.last_metadata.get("source_type") == "LAS":
                curve_meta = self.last_metadata.get("curve_metadata", {})
                depth_meta = curve_meta.get(actual_depth_col.upper(), {})
                unit = depth_meta.get("unit", "")
                if not unit:
                    well_meta = self.last_metadata.get("well_metadata", {})
                    unit = well_meta.get("STRT", {}).get("unit", "")
            
            # Check column name heuristics if still ambiguous
            if not unit:
                col_lower = actual_depth_col.lower()
                if "ft" in col_lower or "feet" in col_lower:
                    unit = "FT"
                elif "_m" in col_lower or "mtr" in col_lower or "meter" in col_lower:
                    unit = "M"

        unit_str = str(unit).strip().upper() if unit else "UNKNOWN"
        is_feet = unit_str in ["FT", "F", "FEET", "'"]

        if is_feet:
            logger.info(f"Converting depth '{actual_depth_col}' from Feet to Meters (* 0.3048)")
            df_out[target_col_name] = df_out[actual_depth_col].astype(float) * self.FEET_TO_METERS
            if actual_depth_col != target_col_name:
                df_out.drop(columns=[actual_depth_col], inplace=True)
        else:
            logger.info(f"Depth assumed in Meters (unit='{unit_str}'). Setting column '{target_col_name}'")
            if actual_depth_col != target_col_name:
                df_out.rename(columns={actual_depth_col: target_col_name}, inplace=True)

        # Ensure sorted monotonically by depth
        df_out = df_out.sort_values(by=target_col_name).reset_index(drop=True)
        return df_out

    def drop_nulls(
        self,
        df: pd.DataFrame,
        custom_null_values: Optional[List[float]] = None,
        subset: Optional[List[str]] = None,
        how: str = "any",
    ) -> pd.DataFrame:
        """
        Identify sentinel null values (e.g., -999.25, -9999.0) and drop NaN/null records.
        
        Parameters:
            df: Input DataFrame.
            custom_null_values: Additional numerical null sentinels to replace with NaN.
            subset: Specific columns to evaluate nulls for (e.g. key logging curves).
            how: 'any' (drop row if any selected column is NaN) or 'all' (drop if all are NaN).
        """
        df_out = df.copy()

        # Combine default and custom null sentinels
        null_vals = set(self.COMMON_NULL_VALUES)
        if custom_null_values:
            null_vals.update(custom_null_values)

        # Include LAS file's specific null sentinel if available
        if "original_null" in self.last_metadata and self.last_metadata["original_null"] is not None:
            try:
                null_vals.add(float(self.last_metadata["original_null"]))
            except (ValueError, TypeError):
                pass

        # Replace null sentinels with np.nan for numeric columns
        numeric_cols = df_out.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            for sentinel in null_vals:
                df_out.loc[np.isclose(df_out[col], sentinel, atol=1e-2, equal_nan=False), col] = np.nan

        # Drop rows with NaNs according to strategy
        initial_len = len(df_out)
        df_cleaned = df_out.dropna(subset=subset, how=how).reset_index(drop=True)
        dropped_count = initial_len - len(df_cleaned)
        logger.info(f"Dropped {dropped_count} rows containing null/NaN values (mode='{how}').")

        return df_cleaned

    def apply_median_filter(
        self,
        df: pd.DataFrame,
        kernel_size: int = 5,
        exclude_columns: Optional[List[str]] = None,
        target_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Apply a rolling 1D median filter using SciPy (scipy.signal.medfilt or scipy.ndimage.median_filter)
        to smooth out high-frequency sensor noise while preserving geological bed boundaries.

        Parameters:
            df: DataFrame containing well log curves.
            kernel_size: Positive odd integer representing filter window length.
            exclude_columns: Columns to skip (depth columns are excluded automatically).
            target_columns: Specific curves to filter. If None, filters all numeric curves.
        """
        if kernel_size % 2 == 0 or kernel_size < 3:
            raise ValueError(f"kernel_size must be an odd integer >= 3, got {kernel_size}")

        df_filtered = df.copy()

        # Default exclusions
        exclusions = set(exclude_columns or [])
        depth_col = self._detect_depth_column(df_filtered)
        if depth_col:
            exclusions.add(depth_col)
        exclusions.update(["DEPTH_M", "DEPT", "DEPTH"])

        # Determine target columns
        if target_columns:
            cols_to_filter = [c for c in target_columns if c in df_filtered.columns]
        else:
            cols_to_filter = [
                c for c in df_filtered.select_dtypes(include=[np.number]).columns
                if c not in exclusions
            ]

        logger.info(f"Applying rolling median filter (kernel_size={kernel_size}) on: {cols_to_filter}")

        for col in cols_to_filter:
            series_data = df_filtered[col].to_numpy(dtype=float)
            
            # Use scipy.ndimage.median_filter which handles boundary modes gracefully
            # and works safely with continuous float arrays
            # For arrays with NaN, we interpolate or use nearest valid slice
            has_nans = np.isnan(series_data).any()
            if has_nans:
                # Interpolate temporarily for filtering, or use pandas rolling median
                valid_mask = ~np.isnan(series_data)
                if np.sum(valid_mask) >= kernel_size:
                    filtered_valid = median_filter(series_data[valid_mask], size=kernel_size, mode="nearest")
                    series_data[valid_mask] = filtered_valid
                    df_filtered[col] = series_data
            else:
                try:
                    df_filtered[col] = medfilt(series_data, kernel_size=kernel_size)
                except Exception:
                    df_filtered[col] = median_filter(series_data, size=kernel_size, mode="nearest")

        return df_filtered

    def parse_and_clean(
        self,
        filepath: Union[str, Path],
        kernel_size: int = 5,
        target_depth_unit: str = "M",
        drop_strategy: str = "any",
        filter_curves: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        High-level pipeline:
        1. Ingest .LAS or .CSV
        2. Standardize depth to meters
        3. Drop null and sentinel values
        4. Apply rolling median filter for sensor noise reduction
        """
        resolved_path = self._resolve_file_path(filepath)
        ext = resolved_path.suffix.lower()

        if ext == ".las":
            df_raw, metadata = self.ingest_las(resolved_path)
        elif ext == ".csv":
            df_raw, metadata = self.ingest_csv(resolved_path)
        else:
            raise ValueError(f"Unsupported file format '{ext}'. Must be .LAS or .CSV.")

        # Standardize depth to meters
        df_depth = self.standardize_depth(df_raw, target_col_name="DEPTH_M")

        # Drop null values
        df_clean = self.drop_nulls(df_depth, how=drop_strategy)

        # Apply SciPy rolling median filter
        df_filtered = self.apply_median_filter(
            df_clean,
            kernel_size=kernel_size,
            target_columns=filter_curves,
        )

        # Explicitly type-cast Depth, WOB, RPM, Gamma Ray, ROP and all numeric columns to float64
        df_hardened = self.enforce_type_casting(df_filtered)

        metadata["processed_rows"] = len(df_hardened)
        metadata["depth_unit"] = "M"
        metadata["median_kernel_size"] = kernel_size

        return df_hardened, metadata

    @staticmethod
    def enforce_type_casting(df: pd.DataFrame) -> pd.DataFrame:
        """
        Hardens data ingestion by explicitly type-casting Depth, WOB, RPM,
        Gamma Ray, and all other petrophysical/drilling curve columns to float64
        before they are written to Streamlit session state or queried by the visualization engine.
        """
        df_cast = df.copy()
        target_curves = [
            "DEPTH_M", "DEPTH", "DEPT", "MD", "TVD", "depth_m", "depth", "dept", "md",
            "WOB", "WEIGHT_ON_BIT", "wob", "weight_on_bit",
            "RPM", "ROTATIONS_PER_MINUTE", "rpm", "rotations_per_minute",
            "GR", "GAMMA_RAY", "GAMMA", "gr", "gamma_ray", "gamma",
            "ROP", "RATE_OF_PENETRATION", "rop", "rate_of_penetration",
            "RES", "RESISTIVITY", "res", "resistivity",
            "X_EAST_M", "Y_NORTH_M", "X", "Y"
        ]

        # Convert matched curves explicitly
        lower_cols = {col.lower(): col for col in df_cast.columns}
        for curve in target_curves:
            c_low = curve.lower()
            if c_low in lower_cols:
                actual_col = lower_cols[c_low]
                try:
                    df_cast[actual_col] = pd.to_numeric(df_cast[actual_col], errors="coerce").astype("float64")
                except Exception as e:
                    logger.warning(f"Could not convert column '{actual_col}' to float64: {e}")

        # Ensure all numeric columns are strictly float64
        for col in df_cast.select_dtypes(include=[np.number]).columns:
            if df_cast[col].dtype != np.float64:
                df_cast[col] = df_cast[col].astype("float64")

        return df_cast

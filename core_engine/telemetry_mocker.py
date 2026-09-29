"""
GeoInsight-RigX (eRTMAC-NWIS)
Core Engine - Live Telemetry Simulation Engine
Provides DrillingSimulator to mock high-frequency surface and downhole drilling streams
(Depth, WOB, RPM, ROP, Gamma Ray) simulating active drill-bit progression.
"""

import time
import os
import math
import logging
from pathlib import Path
from typing import Generator, Dict, Any, Optional, Union
import pandas as pd
import numpy as np

logger = logging.getLogger("DrillingSimulator")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class DrillingSimulator:
    """
    Simulates real-time telemetry streaming from an active drilling rig.
    Reads historical time-series logs from data_source/production/,
    yielding downward advancing depth, TVD, Inclination, WOB, RPM, ROP,
    Gamma Ray, and Pressure telemetry simulating active drill-bit progression.
    """

    DEFAULT_CSV_NAME = "active_rig_telemetry.csv"

    def __init__(self, data_dir: Optional[Union[str, Path]] = None, initial_file: Optional[str] = None):
        """
        Initialize simulator with base directory and initial file.
        Defaults to project data_source/production/.
        """
        if data_dir is not None:
            self.data_dir = Path(data_dir)
        else:
            base_dir = Path(__file__).resolve().parent.parent
            default_telemetry = os.getenv("TELEMETRY_DATA_DIR", str(base_dir / "data_source" / "production"))
            self.data_dir = Path(default_telemetry)

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self._resolve_file_path(initial_file or self.DEFAULT_CSV_NAME)
        self.df: pd.DataFrame = pd.DataFrame()
        self.current_index: int = 0
        self.current_depth: float = 3200.0
        self.is_streaming: bool = False

        self.load_data(self.file_path)

    def _resolve_file_path(self, filepath: Union[str, Path]) -> Path:
        """Resolve file path either as absolute or relative to self.data_dir."""
        p = Path(filepath)
        if p.is_file():
            return p
        candidate = self.data_dir / filepath
        return candidate

    def load_data(self, file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Read historical time-series CSV file from data_source/production/.
        If missing, generates high-fidelity telemetry automatically.
        Ensures Depth, TVD, Inclination, WOB, RPM, ROP, Gamma_Ray, and Pressure exist.
        """
        target_path = self._resolve_file_path(file_path or self.file_path)

        if not target_path.is_file():
            logger.warning(f"Telemetry file {target_path} not found. Synthesizing active rig dataset.")
            self._synthesize_default_telemetry(target_path)

        try:
            self.df = pd.read_csv(target_path)
            # Standardize column naming
            col_map = {}
            for c in self.df.columns:
                cl = c.strip().lower()
                if cl in ["depth", "depth_m", "md"]:
                    col_map[c] = "Depth"
                elif cl in ["tvd", "true_vertical_depth", "tvd_m"]:
                    col_map[c] = "TVD"
                elif cl in ["inclination", "inc", "deviation", "dev"]:
                    col_map[c] = "Inclination"
                elif cl in ["wob", "weight_on_bit"]:
                    col_map[c] = "WOB"
                elif cl in ["rpm", "rotations_per_minute"]:
                    col_map[c] = "RPM"
                elif cl in ["rop", "rate_of_penetration"]:
                    col_map[c] = "ROP"
                elif cl in ["gamma_ray", "gamma", "gr"]:
                    col_map[c] = "Gamma_Ray"
                elif cl in ["pressure", "press", "live_pressure", "spp", "standpipe_pressure", "pressure_psi"]:
                    col_map[c] = "Pressure"
            self.df.rename(columns=col_map, inplace=True)

            # Ensure numeric conversion
            for metric in ["Depth", "TVD", "Inclination", "WOB", "RPM", "ROP", "Gamma_Ray", "Pressure"]:
                if metric in self.df.columns:
                    self.df[metric] = pd.to_numeric(self.df[metric], errors="coerce")

            self.df.dropna(subset=["Depth"], inplace=True)
            self.df.reset_index(drop=True, inplace=True)

            # Backward-compatible graceful derivation if TVD, Inclination, or Pressure are missing
            depths = self.df["Depth"].values
            n_rows = len(depths)

            if "Inclination" not in self.df.columns or self.df["Inclination"].isna().all():
                # Deviated well inclination starting around 17.5 deg at 3200m and gently building
                inc_base = 17.5 + (depths - 3200.0) * 0.010
                inc = inc_base + 0.12 * np.sin(depths / 14.0)
                self.df["Inclination"] = np.round(np.clip(inc, 15.0, 25.0), 2)

            if "TVD" not in self.df.columns or self.df["TVD"].isna().all():
                # Derive TVD dynamically: TVD = sum(MD_increment * cos(inclination))
                incs = self.df["Inclination"].values
                tvd_arr = np.zeros(n_rows)
                # Calibrated baseline starting TVD = 3168.0m at 3200.0m MD (deviated offshore well)
                tvd_arr[0] = 3168.0 + (depths[0] - 3200.0) * math.cos(math.radians(incs[0]))
                for i in range(1, n_rows):
                    d_md = depths[i] - depths[i - 1]
                    inc_rad = math.radians(float(incs[i]))
                    tvd_arr[i] = tvd_arr[i - 1] + d_md * math.cos(inc_rad)
                self.df["TVD"] = np.round(tvd_arr, 2)

            if "Pressure" not in self.df.columns or self.df["Pressure"].isna().all():
                # Realistic drilling pressure: hydrostatic column + circulating friction (ECD) + hazard effects
                tvd = self.df["TVD"].values
                wob = self.df["WOB"].values if "WOB" in self.df.columns else np.full(n_rows, 25.0)
                rop = self.df["ROP"].values if "ROP" in self.df.columns else np.full(n_rows, 20.0)
                rpm = self.df["RPM"].values if "RPM" in self.df.columns else np.full(n_rows, 120.0)

                # Base hydrostatic pressure gradient ~1.35 psi/m (11.2 ppg mud)
                p_hydro = 4268.0 + 1.35 * (tvd - 3168.0)
                # Circulating dynamics correlated with drilling parameters
                p_circ = 0.35 * (wob - 25.0) + 0.22 * (rop - 20.0) + 0.12 * (rpm - 120.0)
                # Small smooth operational variation
                p_fluct = 3.5 * np.sin(depths / 16.0) + 1.8 * np.cos(depths / 7.0)

                # Geohazard zones:
                # 3245m Stuck pipe -> surge in torque/differential
                h_stuck = 55.0 * np.exp(-((depths - 3245.0) / 2.8) ** 2)
                # 3290m Fractured mud loss -> collapse of hydrostatic head / loss of standpipe pressure
                h_loss = -220.0 * np.exp(-((depths - 3290.0) / 2.8) ** 2)
                # 3340m Borehole pack-off -> annular restriction standpipe surge
                h_pack = 310.0 * np.exp(-((depths - 3340.0) / 2.8) ** 2)

                pressure = p_hydro + p_circ + p_fluct + h_stuck + h_loss + h_pack
                self.df["Pressure"] = np.round(np.clip(pressure, 3600.0, 5200.0), 1)

            self.file_path = target_path
            self.current_index = 0
            if not self.df.empty:
                self.current_depth = float(self.df.iloc[0]["Depth"])

            logger.info(f"Loaded {len(self.df)} telemetry frames from {target_path.name}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading telemetry CSV {target_path}: {e}")
            raise

    def _synthesize_default_telemetry(self, output_path: Path, n_records: int = 660) -> None:
        """
        Synthesize default time-series data spanning continuously from 3200.0m
        to 3365.0m (Total Depth TD). Spans across key hazard intervals:
        - 3245m: Stuck Pipe
        - 3290m: Fractured Mud Loss
        - 3340m: Pack-Off
        - 3365m: Target Well Total Depth (End of Run)
        """
        import datetime
        start_time = datetime.datetime.now() - datetime.timedelta(seconds=n_records)
        timestamps = [start_time + datetime.timedelta(seconds=i) for i in range(n_records)]
        start_depth = 3200.0
        final_depth = 3365.0

        base_depths = np.linspace(start_depth, final_depth, n_records)
        noise_d = np.cumsum(np.random.normal(0, 0.02, n_records))
        noise_d -= np.linspace(noise_d[0], noise_d[-1], n_records)
        depths = np.round(base_depths + noise_d, 2)
        depths[-1] = 3365.0

        rop_base = 22.0 + np.sin(depths / 16.0) * 7.5 + np.random.normal(0, 1.2, n_records)
        for i in range(n_records):
            d = depths[i]
            if 3242.0 <= d <= 3248.0:
                rop_base[i] = np.clip(rop_base[i] * 0.45, 6.0, 13.0)
            elif 3288.0 <= d <= 3294.0:
                rop_base[i] = np.clip(rop_base[i] * 1.45, 26.0, 38.0)
            elif 3338.0 <= d <= 3344.0:
                rop_base[i] = np.clip(rop_base[i] * 0.5, 8.0, 14.0)

        rop = np.clip(rop_base, 6.0, 42.0)
        wob = np.clip(26.0 + np.sin(depths / 18.0) * 6.0 + np.random.normal(0, 1.2, n_records), 14.0, 42.0)
        rpm = np.clip(120.0 + np.cos(depths / 22.0) * 15.0 + np.random.normal(0, 2.5, n_records), 80.0, 160.0)
        gr = np.clip(70.0 + np.sin(depths / 25.0) * 35.0 + np.random.normal(0, 2.0, n_records), 20.0, 140.0)

        # Deviated inclination & TVD
        inc = np.clip(17.5 + (depths - 3200.0) * 0.010 + 0.12 * np.sin(depths / 14.0), 15.0, 25.0)
        tvd = np.zeros(n_records)
        tvd[0] = 3168.0 + (depths[0] - 3200.0) * math.cos(math.radians(inc[0]))
        for i in range(1, n_records):
            d_md = depths[i] - depths[i - 1]
            tvd[i] = tvd[i - 1] + d_md * math.cos(math.radians(inc[i]))

        # Realistic drilling pressure
        p_hydro = 4268.0 + 1.35 * (tvd - 3168.0)
        p_circ = 0.35 * (wob - 25.0) + 0.22 * (rop - 20.0) + 0.12 * (rpm - 120.0)
        p_fluct = 3.5 * np.sin(depths / 16.0) + 1.8 * np.cos(depths / 7.0)
        h_stuck = 55.0 * np.exp(-((depths - 3245.0) / 2.8) ** 2)
        h_loss = -220.0 * np.exp(-((depths - 3290.0) / 2.8) ** 2)
        h_pack = 310.0 * np.exp(-((depths - 3340.0) / 2.8) ** 2)
        pressure = np.clip(p_hydro + p_circ + p_fluct + h_stuck + h_loss + h_pack, 3600.0, 5200.0)

        synth_df = pd.DataFrame({
            "Timestamp": [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps],
            "Depth": depths,
            "TVD": np.round(tvd, 2),
            "Inclination": np.round(inc, 2),
            "ROP": np.round(rop, 2),
            "WOB": np.round(wob, 2),
            "RPM": np.round(rpm, 1),
            "Gamma_Ray": np.round(gr, 2),
            "Pressure": np.round(pressure, 1),
        })
        synth_df.to_csv(output_path, index=False)

    def get_current_depth(self) -> float:
        """
        Track and return the mocked downward progression depth in real-time (meters).
        """
        return self.current_depth

    def stream_telemetry(
        self,
        file_path: Optional[Union[str, Path]] = None,
        delay_sec: float = 1.0,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generator function that yields one row of drilling data
        (Depth, WOB, RPM, ROP, Gamma Ray) per second to simulate an active,
        downward-advancing drill bit. Cleanly halts and triggers End-of-Run modal at final CSV row.
        """
        if file_path is not None:
            self.load_data(file_path)

        if self.df.empty:
            raise ValueError("No telemetry data available to stream.")

        self.is_streaming = True
        n_rows = len(self.df)

        while self.is_streaming:
            row = self.df.iloc[self.current_index].to_dict()
            self.current_depth = float(row.get("Depth", self.current_depth))

            telemetry_point = {
                "Timestamp": row.get("Timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
                "Depth": self.current_depth,
                "TVD": float(row.get("TVD", self.current_depth * 0.9898)),
                "Inclination": float(row.get("Inclination", 17.5)),
                "WOB": float(row.get("WOB", 25.0)),
                "RPM": float(row.get("RPM", 120.0)),
                "ROP": float(row.get("ROP", 20.0)),
                "Gamma_Ray": float(row.get("Gamma_Ray", 75.0)),
                "Pressure": float(row.get("Pressure", 4280.0)),
                "frame_index": self.current_index,
            }

            yield telemetry_point

            # Check if reaching final row of telemetry CSV
            if self.current_index >= n_rows - 1:
                logger.info(f"Target depth reached at row {self.current_index + 1}/{n_rows}. Halting telemetry stream.")
                self.is_streaming = False
                self.is_run_completed = True
                try:
                    import streamlit as st
                    st.session_state["run_completed"] = True
                    st.session_state["show_eor_modal"] = True
                    if "active_rig_metrics" in st.session_state:
                        st.session_state.active_rig_metrics["is_streaming"] = False
                        st.session_state.active_rig_metrics["run_completed"] = True
                except Exception:
                    pass
                break

            # Advance index
            self.current_index += 1

            if delay_sec > 0:
                time.sleep(delay_sec)

    def step(self, n: int = 1) -> Dict[str, Any]:
        """
        Single step generator tick for UI state synchronization without blocking sleep.
        Advances current depth and returns the next telemetry frame.
        Cleanly halts and sets run_completed and show_eor_modal when the final row is reached.
        """
        if self.df.empty:
            self.load_data()

        n_rows = len(self.df)
        is_completed = False

        if self.current_index + n >= n_rows - 1:
            self.current_index = n_rows - 1
            self.is_streaming = False
            self.is_run_completed = True
            is_completed = True
            logger.info("Drilling simulator reached final row of CSV. Telemetry halted.")
            try:
                import streamlit as st
                st.session_state["run_completed"] = True
                st.session_state["show_eor_modal"] = True
                if "active_rig_metrics" in st.session_state:
                    st.session_state.active_rig_metrics["is_streaming"] = False
                    st.session_state.active_rig_metrics["run_completed"] = True
            except Exception:
                pass
        else:
            self.current_index += n

        row = self.df.iloc[self.current_index].to_dict()
        self.current_depth = float(row.get("Depth", self.current_depth))

        # Check hazard proximity against offset well critical depth markers
        hazard = self.check_hazard_proximity(self.current_depth)

        return {
            "Timestamp": row.get("Timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
            "Depth": self.current_depth,
            "TVD": float(row.get("TVD", self.current_depth * 0.9898)),
            "Inclination": float(row.get("Inclination", 17.5)),
            "WOB": float(row.get("WOB", 25.0)),
            "RPM": float(row.get("RPM", 120.0)),
            "ROP": float(row.get("ROP", 20.0)),
            "Gamma_Ray": float(row.get("Gamma_Ray", 75.0)),
            "Pressure": float(row.get("Pressure", 4280.0)),
            "frame_index": self.current_index,
            "active_hazard": hazard,
            "is_run_completed": is_completed or getattr(self, "is_run_completed", False),
        }

    def jump_to_final_row(self) -> Dict[str, Any]:
        """
        Advance bit directly to final row of the CSV file.
        Cleanly halts telemetry streaming and triggers End-of-Run modal.
        """
        if self.df.empty:
            self.load_data()

        self.current_index = len(self.df) - 1
        self.is_streaming = False
        self.is_run_completed = True
        row = self.df.iloc[self.current_index].to_dict()
        self.current_depth = float(row.get("Depth", self.FINAL_TARGET_DEPTH))

        hazard = self.check_hazard_proximity(self.current_depth)
        telemetry_point = {
            "Timestamp": row.get("Timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
            "Depth": self.current_depth,
            "TVD": float(row.get("TVD", self.current_depth * 0.9898)),
            "Inclination": float(row.get("Inclination", 19.1)),
            "WOB": float(row.get("WOB", 25.0)),
            "RPM": float(row.get("RPM", 120.0)),
            "ROP": float(row.get("ROP", 20.0)),
            "Gamma_Ray": float(row.get("Gamma_Ray", 75.0)),
            "Pressure": float(row.get("Pressure", 4490.0)),
            "frame_index": self.current_index,
            "active_hazard": hazard,
            "is_run_completed": True,
        }

        try:
            import streamlit as st
            st.session_state["run_completed"] = True
            st.session_state["show_eor_modal"] = True
            if "active_rig_metrics" in st.session_state:
                st.session_state.active_rig_metrics["is_streaming"] = False
                st.session_state.active_rig_metrics["run_completed"] = True
                st.session_state.active_rig_metrics["current_depth"] = self.current_depth
                st.session_state.active_rig_metrics["latest_frame"] = telemetry_point
        except Exception:
            pass

        return telemetry_point

    def get_streamed_history(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Return the history of frames streamed so far up to current_index.
        """
        if self.df.empty:
            return pd.DataFrame()
        
        # If current_index > 0, return slice [0 : current_index + 1]
        end_idx = min(len(self.df), self.current_index + 1)
        sub = self.df.iloc[:end_idx].copy()
        if limit is not None and len(sub) > limit:
            sub = sub.iloc[-limit:]
        return sub

    # Mock Hazard Database for Offset Wells (Events observed during offset drilling)
    MOCK_HAZARDS = [
        {
            "hazard_id": "HAZ-3245",
            "hazard_type": "Stuck Pipe",
            "title": "Differential Stuck Pipe & Mechanical Binding",
            "depth_m": 3245.0,
            "threshold_m": 10.0,
            "severity": "CRITICAL",
            "offset_well": "Volve-12B (RIGX-ALPHA-01)",
            "historical_event": "Offset well Volve-12B experienced catastrophic differential sticking at 3,245 m MD while drilling high-permeability sandstone. BHA became mechanically bound under 450 psi differential overbalance; required 48 hrs jarring and acid wash.",
            "formation": "Heimdal Member Sandstone (High Porosity / Permeable)",
            "historical_npt_hrs": 48.0,
            "financial_impact": "$420,000 NPT Loss",
            "symptoms": "Torque fluctuations (+85%), elevated hook load drag (+35 klbs), erratic RPM drop.",
            "recommended_actions": [
                "Reduce rotary speed (RPM) to 60 RPM immediately to prevent drillstring twist-off.",
                "Decrease WOB to < 14 klbs; maintain continuous pipe reciprocation and rotation.",
                "Spot 50 bbl low-viscosity lubricating pipe-freeing pill sweep across BHA.",
                "Decrease differential pressure and monitor torque trends continuously.",
            ],
            "mitigation_target_rpm": 60,
            "mitigation_target_wob": 14.0,
            "mitigation_mud_weight_delta": -0.1,
            "active_mitigation_status": "PENDING",
        },
        {
            "hazard_id": "HAZ-3290",
            "hazard_type": "Mud Loss",
            "title": "Severe Fractured Formation Mud Loss / Lost Circulation",
            "depth_m": 3290.0,
            "threshold_m": 10.0,
            "severity": "CRITICAL",
            "offset_well": "Volve-12B (RIGX-ALPHA-01)",
            "historical_event": "Offset well Volve-12B experienced severe structural mud loss (350 bbl/hr) at 3,290 m MD upon penetrating micro-fractured carbonate horizon. Hydrostatic head collapsed, risking secondary gas kick.",
            "formation": "Lower Hordaland Fractured Carbonate",
            "historical_npt_hrs": 36.5,
            "financial_impact": "$310,000 Mud Replacement & Tripping",
            "symptoms": "Drop in standpipe pressure (-250 psi), pit level alarm (-18 bbl/min), erratic returns on shale shakers.",
            "recommended_actions": [
                "Immediately decrease mud circulation pump rate to 450 GPM to minimize ECD.",
                "Spot 75 bbl medium-coarse LCM (Lost Circulation Material) pill across fracture zone.",
                "Increase active mud weight by +0.2 ppg incrementally once losses stabilize to sustain borehole walls.",
                "Suspend drilling ahead; monitor dynamic annular pressure sensor (PWD) in real-time.",
            ],
            "mitigation_target_rpm": 75,
            "mitigation_target_wob": 18.0,
            "mitigation_mud_weight_delta": +0.2,
            "active_mitigation_status": "PENDING",
        },
        {
            "hazard_id": "HAZ-3340",
            "hazard_type": "Pack-Off",
            "title": "Borehole Pack-Off & High Cuttings Bed Influx",
            "depth_m": 3340.0,
            "threshold_m": 10.0,
            "severity": "CRITICAL",
            "offset_well": "Volve-12B (RIGX-ALPHA-01)",
            "historical_event": "Offset well Volve-12B suffered borehole pack-off at 3,340 m MD due to reactive shale sloughing and cuttings accumulation in extended reach section.",
            "formation": "Balder Reactive Smectite Shale",
            "historical_npt_hrs": 24.0,
            "financial_impact": "$195,000 Remedial Circulating",
            "symptoms": "Standpipe pressure surge (+400 psi), sudden loss of returns, high back-reaming drag.",
            "recommended_actions": [
                "Stop ROP penetration immediately; pull bit off bottom by 3 meters.",
                "Circulate bottoms-up at maximum allowable annular velocity with tandem high-density viscous pills.",
                "Rotate string at 110 RPM while reciprocating smoothly to agitate cuttings bed.",
            ],
            "mitigation_target_rpm": 110,
            "mitigation_target_wob": 0.0,
            "mitigation_mud_weight_delta": +0.1,
            "active_mitigation_status": "PENDING",
        },
    ]

    FINAL_TARGET_DEPTH = 3360.0

    def check_hazard_proximity(
        self,
        current_depth: float,
        offset_well_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Check if simulated current_depth is within 10 meters of a depth where the
        historical offset well experienced a critical event (e.g. Stuck Pipe, Mud Loss).
        
        If proximity <= 10.0m, triggers a CRITICAL_WARNING flag and pushes it to st.session_state.
        Returns the active hazard dict or None.
        """
        # Determine hazard catalog (use offset_well_data if custom hazards specified, else default)
        hazards = self.MOCK_HAZARDS

        triggered_hazard = None
        for h in hazards:
            distance = abs(current_depth - h["depth_m"])
            if distance <= h.get("threshold_m", 10.0):
                triggered_hazard = dict(h)
                triggered_hazard["distance_to_event"] = round(distance, 1)
                triggered_hazard["delta_m"] = round(current_depth - h["depth_m"], 1)
                break

        # Attempt pushing to Streamlit session_state safely
        try:
            import streamlit as st
            if triggered_hazard is not None:
                st.session_state["critical_warning"] = True
                st.session_state["active_hazard"] = triggered_hazard
                
                # Maintain persistent hazard history
                if "hazard_history" not in st.session_state:
                    st.session_state["hazard_history"] = []
                
                existing_ids = [item["hazard_id"] for item in st.session_state["hazard_history"]]
                if triggered_hazard["hazard_id"] not in existing_ids:
                    st.session_state["hazard_history"].append(triggered_hazard)
                
                logger.warning(
                    f"CRITICAL WARNING: Imminent Hazard Zone detected at {current_depth:.1f} m! "
                    f"Event: {triggered_hazard['title']} (Offset depth: {triggered_hazard['depth_m']} m)"
                )
            else:
                # If no hazard within 10m threshold, clear active warning flag
                st.session_state["critical_warning"] = False
                st.session_state["active_hazard"] = None
        except Exception:
            # Running outside Streamlit context (e.g. testing or CLI)
            pass

        return triggered_hazard

    def is_final_depth_reached(self, current_depth: Optional[float] = None) -> bool:
        """Check if active well has reached or surpassed final target depth."""
        d = current_depth if current_depth is not None else self.current_depth
        return d >= self.FINAL_TARGET_DEPTH

    def reset(self, start_offset: int = 20) -> None:
        """Reset the simulator back to initial depth."""
        self.current_index = min(start_offset, len(self.df) - 1)
        if not self.df.empty:
            self.current_depth = float(self.df.iloc[self.current_index]["Depth"])
        
        # Reset session state hazard flags if available
        try:
            import streamlit as st
            st.session_state["critical_warning"] = False
            st.session_state["active_hazard"] = None
        except Exception:
            pass


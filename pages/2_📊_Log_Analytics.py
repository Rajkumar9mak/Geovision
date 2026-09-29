"""
GeoInsight-RigX (eRTMAC-NWIS)
Page 2: Comparative Log Analytics (The Depth View)
Multi-track petrophysical and drilling telemetry visualizer strictly reading from
st.session_state.selected_offset_wells (populated on Page 1) to dynamically
compare live downhole telemetry against the exact discovered offset well baseline.
"""

import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core_engine.database_manager import SpatialDBManager
from core_engine.visualization_engine import generate_sample_wellbore_data
from core_engine.telemetry_mocker import DrillingSimulator
from core_engine.ui_theme import inject_industrial_theme_css

# Ensure persistent dark industrial theme and Z-index isolation across page switches
inject_industrial_theme_css()

st.markdown("## 📊 Comparative Log Analytics")
st.caption("Subsurface Depth Profile • Live Downhole Stream vs Historical Offset Baseline (eRTMAC-NWIS)")

# ==============================================================================
# EDGE-CASE FALLBACK: VERIFY TARGET WELL SELECTION FROM GEOSPATIAL HUB
# ==============================================================================
offset_state = st.session_state.get("selected_offset_wells", None)

if offset_state is None or "baseline_curves_df" not in offset_state or offset_state["baseline_curves_df"].empty:
    st.warning("⚠️ Awaiting Target Well configuration from the Geospatial Hub.")
    st.info(
        "No active spatial offset baseline is loaded in the session state. "
        "Please initiate the offset search from Page 1, or initialize the default baseline below."
    )
    col_nav1, col_nav2 = st.columns([1, 1.5])
    with col_nav1:
        if st.button("🗺️ Go to Geospatial Hub", key="btn_goto_geohub_p2", use_container_width=True):
            st.switch_page("pages/1_🌍_Geospatial_Hub.py")
    with col_nav2:
        if st.button("⚡ Initialize Default Baseline (Mumbai High Offshore)", key="btn_init_def_p2", use_container_width=True):
            if "db_manager" not in st.session_state:
                st.session_state.db_manager = SpatialDBManager()
            df_off = st.session_state.db_manager.get_nearest_offset_wells(19.4215, 71.3510, radius_km=50.0, limit=3)
            st.session_state.offset_wells = df_off
            
            primary = df_off.iloc[0].to_dict()
            p_name = str(primary.get("well_name", "RIGX-ALPHA-01"))
            p_td = float(primary.get("total_depth_m", 3500.0))
            
            b_df = generate_sample_wellbore_data(well_name=p_name, total_depth_m=p_td, step_m=1.0)
            b_df = b_df[(b_df["DEPTH_M"] >= 3180.0) & (b_df["DEPTH_M"] <= 3450.0)].copy()
            b_df.rename(columns={"DEPTH_M": "Depth", "GR": "Gamma_Ray"}, inplace=True)
            if "RPM" not in b_df.columns:
                b_df["RPM"] = np.clip(115.0 + np.sin(b_df["Depth"] / 20.0) * 12.0, 90.0, 140.0)
            if "TVD" not in b_df.columns:
                b_df["TVD"] = np.round(b_df["Depth"] * 0.9898, 2)
            if "Pressure" not in b_df.columns:
                b_df["Pressure"] = np.round(4250.0 + (b_df["TVD"] - 3168.0) * 1.35, 1)

            st.session_state.selected_offset_wells = {
                "dataframe": df_off,
                "primary_well": primary,
                "primary_well_name": p_name,
                "primary_well_uwi": primary.get("uwi", "IND-OFF-MH-101"),
                "primary_well_distance_km": float(primary.get("distance_km", 0.0)),
                "primary_well_operator": primary.get("operator", "ONGC / GeoInsight"),
                "primary_well_td_m": p_td,
                "baseline_curves_df": b_df,
                "numerical_averages": {
                    "avg_rop": round(float(b_df["ROP"].mean()), 1),
                    "avg_wob": round(float(b_df["WOB"].mean()), 1),
                    "avg_rpm": round(float(b_df["RPM"].mean()), 0),
                    "avg_gr": round(float(b_df["Gamma_Ray"].mean()), 1),
                    "avg_pressure": round(float(b_df["Pressure"].mean()), 1),
                },
                "target_coordinates": {"lat": 19.4215, "lon": 71.3510, "radius_km": 50.0},
            }
            st.rerun()
    st.stop()

# ==============================================================================
# RETRIEVE ACTIVE WELL BASELINE FROM SESSION STATE
# ==============================================================================
primary_name = offset_state["primary_well_name"]
primary_dist = offset_state["primary_well_distance_km"]
primary_uwi = offset_state.get("primary_well_uwi", "N/A")
primary_op = offset_state.get("primary_well_operator", "National Operator")
baseline_df = offset_state["baseline_curves_df"]
averages = offset_state["numerical_averages"]

# Ensure baseline_df has TVD, Pressure, and numerical averages
if "TVD" not in baseline_df.columns:
    baseline_df["TVD"] = np.round(baseline_df["Depth"] * 0.9898, 2)
if "Pressure" not in baseline_df.columns:
    baseline_df["Pressure"] = np.round(4250.0 + (baseline_df["TVD"] - 3168.0) * 1.35, 1)
if "avg_pressure" not in averages:
    averages["avg_pressure"] = round(float(baseline_df["Pressure"].mean()), 1)

# Ensure DrillingSimulator and active_rig_metrics exist
if "drilling_sim" not in st.session_state:
    st.session_state.drilling_sim = DrillingSimulator()
    st.session_state.drilling_sim.reset(start_offset=30)

sim: DrillingSimulator = st.session_state.drilling_sim

if "active_rig_metrics" not in st.session_state:
    hist_initial = sim.get_streamed_history(limit=50)
    st.session_state.active_rig_metrics = {
        "current_depth": sim.get_current_depth(),
        "history_df": hist_initial,
        "latest_frame": hist_initial.iloc[-1].to_dict() if not hist_initial.empty else {},
        "is_streaming": False,
    }

# Helper to switch active baseline directly from discovered wells
def update_offset_baseline_by_index(selected_idx: int):
    offset_df = offset_state["dataframe"]
    target_c = offset_state.get("target_coordinates", {"lat": 19.4215, "lon": 71.3510, "radius_km": 50.0})
    primary = offset_df.iloc[selected_idx].to_dict()
    primary_name = str(primary.get("well_name", "OFFSET-WELL-01"))
    primary_td = max(float(primary.get("total_depth_m", 3500.0)), 3500.0)

    baseline_df = generate_sample_wellbore_data(well_name=primary_name, total_depth_m=primary_td, step_m=1.0)
    baseline_df = baseline_df[(baseline_df["DEPTH_M"] >= 3180.0) & (baseline_df["DEPTH_M"] <= 3450.0)].copy()
    baseline_df.rename(columns={"DEPTH_M": "Depth", "GR": "Gamma_Ray"}, inplace=True)
    if "RPM" not in baseline_df.columns:
        baseline_df["RPM"] = np.clip(115.0 + np.sin(baseline_df["Depth"] / 20.0) * 12.0, 90.0, 140.0)
    if "TVD" not in baseline_df.columns:
        baseline_df["TVD"] = np.round(baseline_df["Depth"] * 0.9898, 2)
    if "Pressure" not in baseline_df.columns:
        baseline_df["Pressure"] = np.round(4250.0 + (baseline_df["TVD"] - 3168.0) * 1.35, 1)

    averages = {
        "avg_rop": round(float(baseline_df["ROP"].mean()), 1),
        "avg_wob": round(float(baseline_df["WOB"].mean()), 1),
        "avg_rpm": round(float(baseline_df["RPM"].mean()), 0),
        "avg_gr": round(float(baseline_df["Gamma_Ray"].mean()), 1),
        "avg_pressure": round(float(baseline_df["Pressure"].mean()), 1),
    }

    st.session_state.selected_offset_wells = {
        "dataframe": offset_df,
        "selected_index": selected_idx,
        "primary_well": primary,
        "primary_well_name": primary_name,
        "primary_well_uwi": primary.get("uwi", "N/A"),
        "primary_well_distance_km": float(primary.get("distance_km", 0.0)),
        "primary_well_operator": primary.get("operator", "National Petroleum Corp"),
        "primary_well_td_m": primary_td,
        "baseline_curves_df": baseline_df,
        "numerical_averages": averages,
        "target_coordinates": target_c,
    }

# Header Control Toolbar
ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.8, 1, 1, 1])

with ctrl_col1:
    discovered_wells = offset_state.get("dataframe", pd.DataFrame())
    if not discovered_wells.empty and len(discovered_wells) > 1:
        well_options = [
            f"{r['well_name']} ({r['distance_km']} km away)"
            for _, r in discovered_wells.iterrows()
        ]
        curr_idx = offset_state.get("selected_index", 0)
        chosen_option = st.selectbox(
            "Benchmark Offset Well (PostGIS)",
            options=range(len(well_options)),
            format_func=lambda i: well_options[i],
            index=min(curr_idx, len(well_options) - 1),
            help="Select among the 3 nearest offset wells discovered via PostGIS on Page 1.",
        )
        if chosen_option != curr_idx:
            update_offset_baseline_by_index(chosen_option)
            st.toast(f"Updated baseline to {well_options[chosen_option]}!", icon="🎯")
            st.rerun()
    else:
        st.markdown(
            f"""
            <div style="font-size:0.88rem; color:#8B949E; padding-top:6px;">
                Target: <b style="color:#E28743;">RIGX-TARGET-01</b> | 
                Offset: <b style="color:#58A6FF;">{primary_name} ({primary_dist} km away)</b><br>
                <span style="font-size:0.75rem; color:#8B949E;">Operator: {primary_op} • UWI: {primary_uwi}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

with ctrl_col2:
    live_toggle = st.toggle("Live Telemetry Stream", value=st.session_state.active_rig_metrics.get("is_streaming", True))
    if live_toggle != st.session_state.active_rig_metrics.get("is_streaming", True):
        st.session_state.active_rig_metrics["is_streaming"] = live_toggle
        st.rerun()

with ctrl_col3:
    if st.button("Advance Bit (+5m)", use_container_width=True, help="Advance simulation step manually"):
        new_point = sim.step(n=5)
        st.session_state.active_rig_metrics["current_depth"] = new_point["Depth"]
        st.session_state.active_rig_metrics["latest_frame"] = new_point
        st.session_state.active_rig_metrics["history_df"] = sim.get_streamed_history(limit=250)
        st.rerun()

with ctrl_col4:
    if st.button("Final Row (TD)", use_container_width=True, help="Simulate bit reaching final row of CSV (3,365m TD)"):
        final_point = sim.jump_to_final_row()
        st.session_state.active_rig_metrics["current_depth"] = final_point["Depth"]
        st.session_state.active_rig_metrics["latest_frame"] = final_point
        st.session_state.active_rig_metrics["history_df"] = sim.get_streamed_history(limit=250)
        st.session_state.active_rig_metrics["is_streaming"] = False
        st.rerun()


# Define Fragment for Streaming Multi-track Chart & Live KPI Cards (Updates every 1s when live)
@st.fragment(run_every="1s" if st.session_state.active_rig_metrics.get("is_streaming", False) else None)
def render_live_telemetry_dashboard():
    # If live streaming is active, step the simulator
    if st.session_state.active_rig_metrics.get("is_streaming", False):
        new_point = sim.step(n=1)
        st.session_state.active_rig_metrics["current_depth"] = new_point["Depth"]
        st.session_state.active_rig_metrics["latest_frame"] = new_point
        st.session_state.active_rig_metrics["history_df"] = sim.get_streamed_history(limit=250)
        if new_point.get("is_run_completed", False):
            st.session_state.active_rig_metrics["is_streaming"] = False
            st.session_state["run_completed"] = True
            st.session_state["show_eor_modal"] = True
            st.rerun()

    live_df = st.session_state.active_rig_metrics["history_df"]
    current_bit_depth = st.session_state.active_rig_metrics["current_depth"]
    latest = st.session_state.active_rig_metrics["latest_frame"]

    # ONE authoritative latest telemetry row (Requirement 16)
    curr_depth = float(latest.get("Depth", current_bit_depth))
    curr_tvd = float(latest.get("TVD", curr_depth * 0.9898))
    curr_inc = float(latest.get("Inclination", 17.6))
    curr_rop = float(latest.get("ROP", 22.4))
    curr_wob = float(latest.get("WOB", 25.1))
    curr_rpm = float(latest.get("RPM", 120.0))
    curr_gr = float(latest.get("Gamma_Ray", 74.5))
    curr_pressure = float(latest.get("Pressure", 4280.0))

    avg_offset_rop = averages.get("avg_rop", 20.0)
    avg_offset_wob = averages.get("avg_wob", 24.0)
    avg_offset_gr = averages.get("avg_gr", 70.0)
    avg_offset_press = averages.get("avg_pressure", 4250.0)

    # TVD delta indicator (Requirement 1)
    if len(live_df) > 1 and "TVD" in live_df.columns:
        prev_tvd = float(live_df.iloc[-2]["TVD"])
        tvd_step = curr_tvd - prev_tvd
    else:
        tvd_step = 0.8
    if tvd_step <= 0.001:
        tvd_step = 0.8
    tvd_indicator = f"↓ {tvd_step:.1f} m/step"

    # Live Pressure delta indicator (Requirement 1)
    delta_press_baseline = curr_pressure - avg_offset_press
    press_arrow = "↑" if delta_press_baseline >= 0 else "↓"
    press_indicator = f"{press_arrow} {delta_press_baseline:+.0f} psi"

    # Depth step delta
    if len(live_df) > 1 and "Depth" in live_df.columns:
        prev_depth = float(live_df.iloc[-2]["Depth"])
        depth_step = curr_depth - prev_depth
    else:
        depth_step = curr_rop / 10.0
    if depth_step <= 0.001:
        depth_step = 0.2

    # Top Live Telemetry KPI Cards (Requirement 1: 7 cards in exact order)
    # Bit Depth (MD), TVD, Live ROP, Live WOB, Rotary Speed (RPM), Live Pressure, Gamma Ray
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6, kpi_col7 = st.columns(7)
    with kpi_col1:
        st.metric("Bit Depth (MD)", f"{curr_depth:.1f} m", f"+{depth_step:.2f} m/step")
    with kpi_col2:
        st.metric("TVD", f"{curr_tvd:.1f} m", tvd_indicator)
    with kpi_col3:
        st.metric("Live ROP", f"{curr_rop:.1f} m/hr", f"{curr_rop - avg_offset_rop:+.1f} vs {primary_name}")
    with kpi_col4:
        st.metric("Live WOB", f"{curr_wob:.1f} klbs", f"{curr_wob - avg_offset_wob:+.1f} vs {primary_name}")
    with kpi_col5:
        st.metric("Rotary Speed (RPM)", f"{curr_rpm:.0f} RPM", "Active Surface Drive")
    with kpi_col6:
        st.metric("Live Pressure", f"{curr_pressure:,.0f} psi", press_indicator)
    with kpi_col7:
        st.metric("Gamma Ray (Lithology)", f"{curr_gr:.1f} API", f"{curr_gr - avg_offset_gr:+.1f} API")

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # Construct High-Density Multi-Track Subplots (Row 1: 3 Columns, Row 2: 2 Columns)
    fig = make_subplots(
        rows=2,
        cols=6,
        specs=[
            [{"colspan": 2}, None, {"colspan": 2}, None, {"colspan": 2}, None],
            [{"colspan": 3}, None, None, {"colspan": 3}, None, None],
        ],
        shared_yaxes=False,
        vertical_spacing=0.09,
        horizontal_spacing=0.035,
        subplot_titles=(
            "<b>Track 1: Rate of Penetration (ROP)</b>",
            "<b>Track 2: Weight on Bit (WOB) & RPM</b>",
            "<b>Track 3: Gamma Ray (Lithology)</b>",
            "<b>Track 4: Formation & Circulating Pressure</b>",
            "<b>Track 5: TVD vs Measured Depth (MD)</b>",
        ),
    )

    # TRACK 1: RATE OF PENETRATION (ROP)
    if "ROP" in baseline_df.columns and "Depth" in baseline_df.columns:
        fig.add_trace(
            go.Scatter(
                x=baseline_df["ROP"],
                y=baseline_df["Depth"],
                mode="lines",
                name=f"{primary_name} Baseline ROP",
                line=dict(color="#58A6FF", width=1.6, dash="dash"),
                opacity=0.65,
                hoverinfo="x+y+name",
            ),
            row=1,
            col=1,
        )

    if not live_df.empty and "ROP" in live_df.columns:
        fig.add_trace(
            go.Scatter(
                x=live_df["ROP"],
                y=live_df["Depth"],
                mode="lines+markers",
                name="Live ROP (Target)",
                line=dict(color="#E28743", width=2.8),
                marker=dict(size=3.5, color="#E28743"),
                hoverinfo="x+y+name",
            ),
            row=1,
            col=1,
        )
        # Highlighted current live point marker (Requirement 6)
        fig.add_trace(
            go.Scatter(
                x=[curr_rop],
                y=[curr_depth],
                mode="markers+text",
                name="Current Live ROP",
                marker=dict(size=10, color="#FFB74D", line=dict(color="#FFFFFF", width=1.6)),
                text=[" LIVE"],
                textposition="middle right",
                textfont=dict(color="#FFB74D", size=9, family="monospace"),
                showlegend=False,
                hoverinfo="x+y+name",
            ),
            row=1,
            col=1,
        )

    # TRACK 2: WEIGHT ON BIT (WOB) & RPM
    if "WOB" in baseline_df.columns:
        fig.add_trace(
            go.Scatter(
                x=baseline_df["WOB"],
                y=baseline_df["Depth"],
                mode="lines",
                name=f"{primary_name} Baseline WOB",
                line=dict(color="#8B949E", width=1.6, dash="dash"),
                opacity=0.6,
                hoverinfo="x+y+name",
            ),
            row=1,
            col=3,
        )

    if "RPM" in baseline_df.columns:
        fig.add_trace(
            go.Scatter(
                x=baseline_df["RPM"] / 5.0,
                y=baseline_df["Depth"],
                mode="lines",
                name=f"{primary_name} Baseline RPM (/5)",
                line=dict(color="#79C0FF", width=1.2, dash="dot"),
                opacity=0.5,
                hoverinfo="x+y+name",
            ),
            row=1,
            col=3,
        )

    if not live_df.empty and "WOB" in live_df.columns:
        fig.add_trace(
            go.Scatter(
                x=live_df["WOB"],
                y=live_df["Depth"],
                mode="lines+markers",
                name="Live WOB (Target)",
                line=dict(color="#E28743", width=2.8),
                marker=dict(size=3.5, color="#E28743"),
                hoverinfo="x+y+name",
            ),
            row=1,
            col=3,
        )

    if not live_df.empty and "RPM" in live_df.columns:
        fig.add_trace(
            go.Scatter(
                x=live_df["RPM"] / 5.0,
                y=live_df["Depth"],
                mode="lines",
                name="Live RPM (/5)",
                line=dict(color="#FFB74D", width=2.2, dash="dot"),
                hoverinfo="x+y+name",
            ),
            row=1,
            col=3,
        )

    if not live_df.empty:
        # Highlighted current live point marker (Requirement 6)
        fig.add_trace(
            go.Scatter(
                x=[curr_wob],
                y=[curr_depth],
                mode="markers+text",
                name="Current Live WOB",
                marker=dict(size=10, color="#FFB74D", line=dict(color="#FFFFFF", width=1.6)),
                text=[" LIVE"],
                textposition="middle right",
                textfont=dict(color="#FFB74D", size=9, family="monospace"),
                showlegend=False,
                hoverinfo="x+y+name",
            ),
            row=1,
            col=3,
        )

    # TRACK 3: GAMMA RAY (LITHOLOGY)
    if "Gamma_Ray" in baseline_df.columns:
        fig.add_trace(
            go.Scatter(
                x=baseline_df["Gamma_Ray"],
                y=baseline_df["Depth"],
                mode="lines",
                name=f"{primary_name} Baseline GR",
                line=dict(color="#6E7681", width=1.6, dash="dash"),
                opacity=0.65,
                hoverinfo="x+y+name",
            ),
            row=1,
            col=5,
        )

    if not live_df.empty and "Gamma_Ray" in live_df.columns:
        fig.add_trace(
            go.Scatter(
                x=live_df["Gamma_Ray"],
                y=live_df["Depth"],
                mode="lines+markers",
                name="Live Gamma Ray (Target)",
                line=dict(color="#E28743", width=2.8),
                marker=dict(size=3.5, color="#E28743"),
                hoverinfo="x+y+name",
            ),
            row=1,
            col=5,
        )
        # Highlighted current live point marker (Requirement 6)
        fig.add_trace(
            go.Scatter(
                x=[curr_gr],
                y=[curr_depth],
                mode="markers+text",
                name="Current Live GR",
                marker=dict(size=10, color="#FFB74D", line=dict(color="#FFFFFF", width=1.6)),
                text=[" LIVE"],
                textposition="middle right",
                textfont=dict(color="#FFB74D", size=9, family="monospace"),
                showlegend=False,
                hoverinfo="x+y+name",
            ),
            row=1,
            col=5,
        )

    # TRACK 4: PRESSURE (Requirement 4)
    if "Pressure" in baseline_df.columns:
        fig.add_trace(
            go.Scatter(
                x=baseline_df["Pressure"],
                y=baseline_df["Depth"],
                mode="lines",
                name=f"{primary_name} Baseline Pressure",
                line=dict(color="#58A6FF", width=1.6, dash="dash"),
                opacity=0.65,
                hoverinfo="x+y+name",
            ),
            row=2,
            col=1,
        )

    if not live_df.empty and "Pressure" in live_df.columns:
        fig.add_trace(
            go.Scatter(
                x=live_df["Pressure"],
                y=live_df["Depth"],
                mode="lines+markers",
                name="Live Pressure (Target)",
                line=dict(color="#E28743", width=2.8),
                marker=dict(size=3.5, color="#E28743"),
                hoverinfo="x+y+name",
            ),
            row=2,
            col=1,
        )
        # Highlighted current live point marker (Requirement 6)
        fig.add_trace(
            go.Scatter(
                x=[curr_pressure],
                y=[curr_depth],
                mode="markers+text",
                name="Current Live Pressure",
                marker=dict(size=10, color="#FFB74D", line=dict(color="#FFFFFF", width=1.6)),
                text=[" LIVE"],
                textposition="middle right",
                textfont=dict(color="#FFB74D", size=9, family="monospace"),
                showlegend=False,
                hoverinfo="x+y+name",
            ),
            row=2,
            col=1,
        )

    # TRACK 5: TVD vs MD (Requirement 5)
    if "TVD" in baseline_df.columns:
        fig.add_trace(
            go.Scatter(
                x=baseline_df["TVD"],
                y=baseline_df["Depth"],
                mode="lines",
                name=f"{primary_name} Baseline TVD",
                line=dict(color="#58A6FF", width=1.6, dash="dash"),
                opacity=0.65,
                hoverinfo="x+y+name",
            ),
            row=2,
            col=4,
        )

    if not live_df.empty and "TVD" in live_df.columns:
        # Reference Vertical Line (MD = TVD) to visually show well deviation
        fig.add_trace(
            go.Scatter(
                x=live_df["Depth"],
                y=live_df["Depth"],
                mode="lines",
                name="Vertical Well Ref (MD=TVD)",
                line=dict(color="#484F58", width=1.2, dash="dot"),
                opacity=0.55,
                hoverinfo="x+y+name",
            ),
            row=2,
            col=4,
        )
        fig.add_trace(
            go.Scatter(
                x=live_df["TVD"],
                y=live_df["Depth"],
                mode="lines+markers",
                name="Live TVD (Target)",
                line=dict(color="#E28743", width=2.8),
                marker=dict(size=3.5, color="#E28743"),
                hoverinfo="x+y+name",
            ),
            row=2,
            col=4,
        )
        # Highlighted current live point marker (Requirement 6)
        fig.add_trace(
            go.Scatter(
                x=[curr_tvd],
                y=[curr_depth],
                mode="markers+text",
                name="Current Live TVD",
                marker=dict(size=10, color="#FFB74D", line=dict(color="#FFFFFF", width=1.6)),
                text=[" LIVE"],
                textposition="middle right",
                textfont=dict(color="#FFB74D", size=9, family="monospace"),
                showlegend=False,
                hoverinfo="x+y+name",
            ),
            row=2,
            col=4,
        )

    # Add Horizontal Current Bit Depth Indicator Line across all 5 tracks (Requirement 7)
    for r, c in [(1, 1), (1, 3), (1, 5), (2, 1), (2, 4)]:
        annot = f"LIVE • {curr_depth:.1f} m MD" if (r, c) in [(1, 5), (2, 4)] else None
        fig.add_hline(
            y=curr_depth,
            line=dict(color="#E28743", width=1.8, dash="dash"),
            annotation_text=annot,
            annotation_position="bottom right",
            annotation_font=dict(color="#E28743", size=10, family="monospace"),
            row=r,
            col=c,
        )

    # Dark Industrial Theme Configuration & Inverted Y-Axis
    min_d = 3190.0
    max_d = max(3380.0, curr_depth + 40.0)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6EDF3", family="sans-serif"),
        height=880,
        margin=dict(l=65, r=30, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(14, 17, 23, 0.8)",
            bordercolor="#30363D",
            borderwidth=1,
            font=dict(size=10, color="#E6EDF3"),
        ),
    )

    # Inverted Y-axes across all tracks
    for r, c in [(1, 1), (1, 3), (1, 5), (2, 1), (2, 4)]:
        is_first_in_row = c == 1
        fig.update_yaxes(
            autorange="reversed",
            range=[max_d, min_d],
            gridcolor="#30363D",
            zerolinecolor="#30363D",
            tickfont=dict(color="#8B949E", size=11),
            title_text="Measured Depth (m MD)" if is_first_in_row else "",
            row=r,
            col=c,
        )

    # X-axes ranges and titles
    fig.update_xaxes(title_text="ROP (m/hr)", range=[0, 45], gridcolor="#30363D", zerolinecolor="#30363D", tickfont=dict(color="#8B949E", size=10), row=1, col=1)
    fig.update_xaxes(title_text="WOB (klbs) / RPM(/5)", range=[0, 50], gridcolor="#30363D", zerolinecolor="#30363D", tickfont=dict(color="#8B949E", size=10), row=1, col=3)
    fig.update_xaxes(title_text="Gamma Ray (API)", range=[15, 160], gridcolor="#30363D", zerolinecolor="#30363D", tickfont=dict(color="#8B949E", size=10), row=1, col=5)
    fig.update_xaxes(title_text="Pressure (psi)", range=[3800, 4800], gridcolor="#30363D", zerolinecolor="#30363D", tickfont=dict(color="#8B949E", size=10), row=2, col=1)
    fig.update_xaxes(title_text="TVD (m)", range=[3140, 3360], gridcolor="#30363D", zerolinecolor="#30363D", tickfont=dict(color="#8B949E", size=10), row=2, col=4)

    st.plotly_chart(fig, use_container_width=True)

# Render live telemetry dashboard
render_live_telemetry_dashboard()

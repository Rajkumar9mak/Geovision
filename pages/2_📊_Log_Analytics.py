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
            b_df["RPM"] = np.clip(115.0 + np.sin(b_df["Depth"] / 20.0) * 12.0, 90.0, 140.0)
            
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

    averages = {
        "avg_rop": round(float(baseline_df["ROP"].mean()), 1),
        "avg_wob": round(float(baseline_df["WOB"].mean()), 1),
        "avg_rpm": round(float(baseline_df["RPM"].mean()), 0),
        "avg_gr": round(float(baseline_df["Gamma_Ray"].mean()), 1),
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
    st.session_state.active_rig_metrics["is_streaming"] = live_toggle

with ctrl_col3:
    if st.button("Advance Bit (+5m)", use_container_width=True, help="Advance simulation step manually"):
        new_point = sim.step(n=5)
        st.session_state.active_rig_metrics["current_depth"] = new_point["Depth"]
        st.session_state.active_rig_metrics["latest_frame"] = new_point
        st.session_state.active_rig_metrics["history_df"] = sim.get_streamed_history(limit=250)
        st.rerun()

with ctrl_col4:
    if st.button("Final Row (TD)", use_container_width=True, help="Simulate bit reaching final row of CSV (3,365m TD)"):
        sim.jump_to_final_row()
        st.rerun()

# Top Live Telemetry KPI Cards compared against numerical baseline averages
latest = st.session_state.active_rig_metrics["latest_frame"]
curr_depth = st.session_state.active_rig_metrics["current_depth"]
curr_rop = latest.get("ROP", 22.4)
curr_wob = latest.get("WOB", 25.1)
curr_rpm = latest.get("RPM", 120.0)
curr_gr = latest.get("Gamma_Ray", 74.5)

avg_offset_rop = averages.get("avg_rop", 20.0)
avg_offset_wob = averages.get("avg_wob", 24.0)
avg_offset_gr = averages.get("avg_gr", 70.0)

m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
with m_col1:
    st.metric("Bit Depth (MD)", f"{curr_depth:.1f} m", f"+{curr_rop/10.0:.2f} m/step")
with m_col2:
    st.metric("Live ROP", f"{curr_rop:.1f} m/hr", f"{curr_rop - avg_offset_rop:+.1f} vs {primary_name}")
with m_col3:
    st.metric("Live WOB", f"{curr_wob:.1f} klbs", f"{curr_wob - avg_offset_wob:+.1f} vs {primary_name}")
with m_col4:
    st.metric("Rotary Speed (RPM)", f"{curr_rpm:.0f} RPM", "Active Surface Drive")
with m_col5:
    st.metric("Gamma Ray (Lithology)", f"{curr_gr:.1f} API", f"{curr_gr - avg_offset_gr:+.1f} API")


# Define Fragment for Streaming Multi-track Chart (Updates every 1s when live)
@st.fragment(run_every="1s" if st.session_state.active_rig_metrics.get("is_streaming", False) else None)
def render_live_multi_track():
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

    # Construct High-Density Multi-Track Subplots (1 Row, 3 Columns)
    fig = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        horizontal_spacing=0.04,
        subplot_titles=(
            "<b>Track 1: Rate of Penetration (ROP)</b>",
            "<b>Track 2: Weight on Bit (WOB) & RPM</b>",
            "<b>Track 3: Gamma Ray (Lithology)</b>",
        ),
    )

    # TRACK 1: RATE OF PENETRATION (ROP)
    # 1. Historical Baseline ROP from discovered offset well (Muted Blue/Grey)
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

    # 2. Live Telemetry ROP (Primary Highlight #E28743)
    if not live_df.empty and "ROP" in live_df.columns:
        fig.add_trace(
            go.Scatter(
                x=live_df["ROP"],
                y=live_df["Depth"],
                mode="lines+markers",
                name="Live ROP (Target)",
                line=dict(color="#E28743", width=2.8),
                marker=dict(size=4, color="#E28743"),
                hoverinfo="x+y+name",
            ),
            row=1,
            col=1,
        )

    # TRACK 2: WEIGHT ON BIT (WOB) & RPM
    # 1. Historical Baseline WOB (Muted Grey)
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
            col=2,
        )

    # 2. Historical Baseline RPM (Muted Blue)
    if "RPM" in baseline_df.columns:
        fig.add_trace(
            go.Scatter(
                x=baseline_df["RPM"] / 5.0,  # Scaled for 0-40 track
                y=baseline_df["Depth"],
                mode="lines",
                name=f"{primary_name} Baseline RPM (/5)",
                line=dict(color="#79C0FF", width=1.2, dash="dot"),
                opacity=0.5,
                hoverinfo="x+y+name",
            ),
            row=1,
            col=2,
        )

    # 3. Live Telemetry WOB (Primary Amber #E28743)
    if not live_df.empty and "WOB" in live_df.columns:
        fig.add_trace(
            go.Scatter(
                x=live_df["WOB"],
                y=live_df["Depth"],
                mode="lines+markers",
                name="Live WOB (Target)",
                line=dict(color="#E28743", width=2.8),
                marker=dict(size=4, color="#E28743"),
                hoverinfo="x+y+name",
            ),
            row=1,
            col=2,
        )

    # 4. Live Telemetry RPM (Warm Gold #FFB74D)
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
            col=2,
        )

    # TRACK 3: GAMMA RAY (LITHOLOGY)
    # 1. Historical Baseline Gamma Ray (Muted Blue/Grey)
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
            col=3,
        )

    # 2. Live Telemetry Gamma Ray (Primary Amber #E28743)
    if not live_df.empty and "Gamma_Ray" in live_df.columns:
        fig.add_trace(
            go.Scatter(
                x=live_df["Gamma_Ray"],
                y=live_df["Depth"],
                mode="lines+markers",
                name="Live Gamma Ray (Target)",
                line=dict(color="#E28743", width=2.8),
                marker=dict(size=4, color="#E28743"),
                hoverinfo="x+y+name",
            ),
            row=1,
            col=3,
        )

    # Add Horizontal Current Bit Depth Indicator Line across all 3 tracks
    for col_idx in [1, 2, 3]:
        fig.add_hline(
            y=current_bit_depth,
            line=dict(color="#E28743", width=1.8, dash="dash"),
            annotation_text=f"Bit: {current_bit_depth:.1f} m" if col_idx == 3 else None,
            annotation_position="bottom right",
            annotation_font=dict(color="#E28743", size=10, family="monospace"),
            row=1,
            col=col_idx,
        )

    # Dark Industrial Theme Configuration & Inverted Y-Axis
    min_d = 3190.0
    max_d = max(3380.0, current_bit_depth + 40.0)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6EDF3", family="sans-serif"),
        height=720,
        margin=dict(l=60, r=30, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(14, 17, 23, 0.8)",
            bordercolor="#30363D",
            borderwidth=1,
            font=dict(size=10, color="#E6EDF3"),
        ),
    )

    # Strictly inverted Y-axis across all tracks (depth increases downwards)
    fig.update_yaxes(
        autorange="reversed",
        title_text="Measured Depth (m MD)",
        range=[max_d, min_d],
        gridcolor="#30363D",
        zerolinecolor="#30363D",
        tickfont=dict(color="#8B949E", size=11),
        row=1,
        col=1,
    )
    fig.update_yaxes(autorange="reversed", gridcolor="#30363D", zerolinecolor="#30363D", row=1, col=2)
    fig.update_yaxes(autorange="reversed", gridcolor="#30363D", zerolinecolor="#30363D", row=1, col=3)

    # Customize X-axes ranges and titles for each track
    fig.update_xaxes(
        title_text="ROP (m/hr)",
        range=[0, 45],
        gridcolor="#30363D",
        zerolinecolor="#30363D",
        tickfont=dict(color="#8B949E", size=10),
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="WOB (klbs) / RPM(/5)",
        range=[0, 50],
        gridcolor="#30363D",
        zerolinecolor="#30363D",
        tickfont=dict(color="#8B949E", size=10),
        row=1,
        col=2,
    )
    fig.update_xaxes(
        title_text="Gamma Ray (API)",
        range=[15, 160],
        gridcolor="#30363D",
        zerolinecolor="#30363D",
        tickfont=dict(color="#8B949E", size=10),
        row=1,
        col=3,
    )

    st.plotly_chart(fig, use_container_width=True)

# Render the multi-track component
render_live_multi_track()

"""
GeoInsight-RigX (eRTMAC-NWIS)
Page 3: Hazard Advisories & Mitigation Control Room
Real-time structural failure diagnostics comparing active formation risks
against offset historical well failures (Volve-12B / RIGX-ALPHA-01)
with actionable mitigation directives and automated End-of-Run AAR summary.
"""

import textwrap
import streamlit as st
import pandas as pd
import numpy as np

from core_engine.telemetry_mocker import DrillingSimulator
from core_engine.ui_theme import inject_industrial_theme_css

# Ensure persistent dark industrial theme and Z-index isolation across page switches
inject_industrial_theme_css()

st.markdown("## ⚠️ Hazard Advisories & Mitigation Control Room")
st.caption("Active Formation Geohazard Diagnostics • Offset Failure Analysis • Precision Mitigation Directives")

# ==============================================================================
# EDGE-CASE FALLBACK: VERIFY TARGET WELL SELECTION FROM GEOSPATIAL HUB
# ==============================================================================
offset_state = st.session_state.get("selected_offset_wells", None)
if offset_state is None:
    st.warning("⚠️ Awaiting Target Well configuration from the Geospatial Hub.")
    st.info(
        "No active spatial offset baseline is loaded in the session state. "
        "Please initiate the offset search from Page 1, or initialize the default baseline below."
    )
    col_nav1, col_nav2 = st.columns([1, 1.5])
    with col_nav1:
        if st.button("🗺️ Go to Geospatial Hub", key="btn_goto_geohub_p3", use_container_width=True):
            st.switch_page("pages/1_🌍_Geospatial_Hub.py")
    with col_nav2:
        if st.button("⚡ Initialize Default Baseline (Mumbai High Offshore)", key="btn_init_def_p3", use_container_width=True):
            from core_engine.database_manager import SpatialDBManager
            from core_engine.visualization_engine import generate_sample_wellbore_data
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

# Ensure simulator and metrics exist
if "drilling_sim" not in st.session_state:
    st.session_state.drilling_sim = DrillingSimulator()

sim: DrillingSimulator = st.session_state.drilling_sim

# Active depth and hazard retrieval
current_depth = st.session_state.get("active_rig_metrics", {}).get("current_depth", sim.get_current_depth())

# Re-check proximity to ensure state is synchronized
active_hazard = sim.check_hazard_proximity(current_depth)
hazard_history = st.session_state.get("hazard_history", [])

# Simulation Control Bar for Geohazard Proximity & Final Row Testing
with st.container():
    c_tools_1, c_tools_2, c_tools_3, c_tools_4, c_tools_5 = st.columns([1.5, 1, 1, 1, 1])
    with c_tools_1:
        st.markdown(
            f"""
            <div style="font-size:0.85rem; color:#8B949E; padding-top:6px;">
                Active Bit Depth: <b style="color:#E28743;">{current_depth:.1f} m MD</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_tools_2:
        if st.button("Stuck Pipe (3,245m)", use_container_width=True, help="Simulate bit entering 3,245m Stuck Pipe zone"):
            sim.current_depth = 3245.0
            if "active_rig_metrics" in st.session_state:
                st.session_state.active_rig_metrics["current_depth"] = 3245.0
            sim.check_hazard_proximity(3245.0)
            st.rerun()
    with c_tools_3:
        if st.button("Mud Loss (3,290m)", use_container_width=True, help="Simulate bit entering 3,290m Mud Loss zone"):
            sim.current_depth = 3290.0
            if "active_rig_metrics" in st.session_state:
                st.session_state.active_rig_metrics["current_depth"] = 3290.0
            sim.check_hazard_proximity(3290.0)
            st.rerun()
    with c_tools_4:
        if st.button("Pack-Off (3,340m)", use_container_width=True, help="Simulate bit entering 3,340m Borehole Pack-Off zone"):
            sim.current_depth = 3340.0
            if "active_rig_metrics" in st.session_state:
                st.session_state.active_rig_metrics["current_depth"] = 3340.0
            sim.check_hazard_proximity(3340.0)
            st.rerun()
    with c_tools_5:
        if st.button("Final Row (TD)", use_container_width=True, help="Simulate reaching final row of CSV (3,365m TD) and trigger modal"):
            sim.jump_to_final_row()
            st.session_state["show_eor_modal"] = True
            st.rerun()

st.divider()

# Condition: Page must ONLY display hazard details if a hazard is currently triggered or historically flagged
has_active_hazard = (active_hazard is not None) or st.session_state.get("critical_warning", False)
has_historical_hazards = len(hazard_history) > 0

if not has_active_hazard and not has_historical_hazards:
    # All Clear / Green Zone Standby Display
    st.markdown(
        """
        <div style="background-color:#161B22; border:1px solid #30363D; border-radius:8px; padding:30px; text-align:center; margin:20px 0;">
            <div style="font-size:2.5rem; margin-bottom:10px;">🟢</div>
            <h3 style="color:#4CAF50; margin-bottom:6px;">GREEN FORMATION ZONE — NO ACTIVE HAZARDS</h3>
            <p style="color:#8B949E; max-width:650px; margin:0 auto; font-size:0.95rem;">
                Continuous subsurface horizon scanning is active. The current bit position is outside historical hazard threshold zones (< 10 m).
                Telemetry values remain within safe drilling parameters. Use the control bar above to test proximity triggers.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # Select which hazard to display: active hazard prioritized, otherwise latest in history
    display_hazard = active_hazard if active_hazard is not None else hazard_history[-1]

    # Active Hazard Alert Banner
    delta_str = f"{display_hazard.get('delta_m', 0.0):+.1f} m"
    dist_str = f"{display_hazard.get('distance_to_event', 0.0):.1f} m"
    is_live_now = active_hazard is not None and active_hazard.get("hazard_id") == display_hazard.get("hazard_id")

    banner_bg = "#8B0000" if is_live_now else "#1F242D"
    banner_border = "#FF4444" if is_live_now else "#E28743"
    tag_label = "CRITICAL WARNING • IMMINENT HAZARD ZONE" if is_live_now else "HISTORICALLY FLAGGED HAZARD ZONE"

    st.markdown(
        f"""
        <div style="background-color:{banner_bg}; border:2px solid {banner_border}; border-radius:8px; padding:16px 20px; margin-bottom:20px; box-shadow:0 0 16px rgba(255,68,68,0.35);">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="background:rgba(255,255,255,0.2); color:#FFFFFF; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:800; letter-spacing:0.05em;">
                        {tag_label}
                    </span>
                    <h3 style="color:#FFFFFF; margin:8px 0 4px 0; font-size:1.35rem;">
                        {display_hazard['title']}
                    </h3>
                    <div style="color:#FFCDD2; font-size:0.85rem;">
                        Formation: <b>{display_hazard.get('formation', 'Subsurface Stratum')}</b> | 
                        Target Hazard Depth: <b>{display_hazard['depth_m']:.1f} m MD</b> | 
                        Distance from Bit: <b>{dist_str}</b> ({delta_str})
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:0.75rem; color:#FFCDD2;">HISTORICAL LOSS</div>
                    <div style="font-size:1.15rem; font-weight:800; color:#FFFFFF;">{display_hazard.get('financial_impact', '$350k')}</div>
                    <div style="font-size:0.75rem; color:#FFCDD2;">NPT: {display_hazard.get('historical_npt_hrs', 36.0)} hrs</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # High-Density Dual-Column Control Room Layout
    col_hist, col_mitigation = st.columns([1, 1], gap="large")

    # ==============================================================================
    # LEFT COLUMN: HISTORICAL OFFSET FAILURE DATA
    # ==============================================================================
    with col_hist:
        st.markdown("### 📜 Offset Well Failure Analysis")
        st.markdown(
            f"""
            <div style="background-color:#161B22; border:1px solid #30363D; border-radius:8px; padding:18px; margin-bottom:15px;">
                <div style="font-size:0.8rem; color:#8B949E; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">
                    Benchmark Reference Well
                </div>
                <div style="font-size:1.1rem; font-weight:700; color:#58A6FF; margin-bottom:12px;">
                    {display_hazard.get('offset_well', 'Volve-12B / RIGX-ALPHA-01')}
                </div>
                <div style="font-size:0.88rem; color:#E6EDF3; line-height:1.55; margin-bottom:14px; background:#0E1117; padding:12px; border-left:3px solid #FF4444; border-radius:4px;">
                    "{display_hazard.get('historical_event', 'Catastrophic event observed during drilling.')}"
                </div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; font-size:0.82rem; color:#8B949E; margin-top:10px;">
                    <div style="background:#1F242D; padding:8px 12px; border-radius:5px;">
                        <span>Event Depth:</span><br>
                        <b style="color:#F0F6FC; font-size:0.95rem;">{display_hazard['depth_m']:.1f} m MD</b>
                    </div>
                    <div style="background:#1F242D; padding:8px 12px; border-radius:5px;">
                        <span>Non-Productive Time:</span><br>
                        <b style="color:#FF7B72; font-size:0.95rem;">{display_hazard.get('historical_npt_hrs', 0)} Hours NPT</b>
                    </div>
                    <div style="background:#1F242D; padding:8px 12px; border-radius:5px;">
                        <span>Financial Burden:</span><br>
                        <b style="color:#FF7B72; font-size:0.95rem;">{display_hazard.get('financial_impact', 'N/A')}</b>
                    </div>
                    <div style="background:#1F242D; padding:8px 12px; border-radius:5px;">
                        <span>Lithology:</span><br>
                        <b style="color:#A5D6A7; font-size:0.85rem;">{display_hazard.get('formation', 'Sandstone')}</b>
                    </div>
                </div>
                <div style="margin-top:14px; font-size:0.82rem; color:#8B949E;">
                    <b>Reported Downhole Symptoms:</b><br>
                    <span style="color:#D2A8FF;">{display_hazard.get('symptoms', 'Torque spikes, high drag, pressure fluctuations')}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ==============================================================================
    # RIGHT COLUMN: ACTIONABLE STRUCTURAL MITIGATION DIRECTIVES
    # ==============================================================================
    with col_mitigation:
        st.markdown("### 🛡️ Actionable Mitigation Directives")

        mitigation_status = display_hazard.get("active_mitigation_status", "PENDING")
        is_mitigated = mitigation_status == "DEPLOYED"

        status_color = "#4CAF50" if is_mitigated else "#FFA000"
        status_text = "PROTOCOLS ACTIVE & DEPLOYED" if is_mitigated else "MITIGATION PROTOCOL PENDING DISPATCH"

        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px; font-size:0.82rem; color:#8B949E;">
                <span style="height:10px; width:10px; background-color:{status_color}; border-radius:50%; display:inline-block;"></span>
                <span>Dispatch Status: <b style="color:{status_color};">{status_text}</b></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Instructions List
        actions = display_hazard.get("recommended_actions", [])
        for i, act in enumerate(actions, 1):
            st.markdown(
                f"""
                <div style="background-color:#161B22; border:1px solid #30363D; border-left:4px solid #E28743; border-radius:6px; padding:12px 14px; margin-bottom:10px;">
                    <div style="font-size:0.75rem; font-weight:700; color:#E28743; text-transform:uppercase;">
                        Action Directive #{i}
                    </div>
                    <div style="font-size:0.9rem; color:#F0F6FC; margin-top:3px; font-weight:500;">
                        {act}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Recommended Parameter Adjustments
        st.markdown(
            f"""
            <div style="background:#1F242D; border:1px solid #30363D; border-radius:6px; padding:12px; margin-bottom:16px; font-size:0.82rem;">
                <div style="color:#8B949E; margin-bottom:6px; font-weight:600;">RECOMMENDED MACHINE OVERRIDES:</div>
                <div style="display:flex; justify-content:space-between;">
                    <span>Target RPM Limit: <b style="color:#E28743;">{display_hazard.get('mitigation_target_rpm', 60)} RPM</b></span>
                    <span>Target WOB Limit: <b style="color:#E28743;">{display_hazard.get('mitigation_target_wob', 14.0)} klbs</b></span>
                    <span>Mud Weight Adj: <b style="color:#E28743;">{display_hazard.get('mitigation_mud_weight_delta', 0.0):+.1f} ppg</b></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Interactive Mitigation Execution Button
        btn_label = "✅ Mitigation Protocols Deployed" if is_mitigated else "🚨 Deploy Structural Mitigation Protocols"
        if st.button(btn_label, use_container_width=True, key="btn_deploy_mitigation"):
            display_hazard["active_mitigation_status"] = "DEPLOYED"
            # Update telemetry values in active stream
            if "active_rig_metrics" in st.session_state:
                latest_f = st.session_state.active_rig_metrics.get("latest_frame", {})
                latest_f["RPM"] = float(display_hazard.get("mitigation_target_rpm", 60))
                latest_f["WOB"] = float(display_hazard.get("mitigation_target_wob", 14.0))
                st.session_state.active_rig_metrics["latest_frame"] = latest_f

            st.toast("Mitigation directives deployed to Rig Floor and Mud Logging Cabin!", icon="🛡️")
            st.rerun()

st.divider()

# ==============================================================================
# OBJECTIVE 4: POST-ACTION SUMMARY EXPANDER & MODAL (END-OF-RUN WELL COMPARISON)
# ==============================================================================
# Native Streamlit Expander for End-of-Run Evaluation (switched from @st.dialog to eliminate layout collisions)
def render_end_of_run_dialog():
    with st.expander("📋 End-of-Run Well Comparison Summary (AAR)", expanded=True):
        st.markdown(
            textwrap.dedent("""
                <div style="background-color:#161B22; border:1px solid #4CAF50; border-radius:8px; padding:16px 20px; margin-bottom:16px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="background:rgba(76,175,80,0.25); color:#4CAF50; padding:3px 10px; border-radius:4px; font-size:0.78rem; font-weight:800; letter-spacing:0.06em;">
                                WELL DRILLING MISSION COMPLETED
                            </span>
                            <h3 style="color:#FFFFFF; margin:8px 0 2px 0;">Target Depth Reached: 3,365.0 m MD</h3>
                            <div style="color:#8B949E; font-size:0.85rem;">Rig: <b>RIGX-TARGET-01</b> • Offset Benchmark: <b>Volve-12B / RIGX-ALPHA-01</b></div>
                        </div>
                        <div style="display:flex; gap:24px; text-align:right;">
                            <div>
                                <div style="font-size:0.75rem; color:#8B949E; text-transform:uppercase;">Total NPT Saved</div>
                                <div style="font-size:1.4rem; font-weight:800; color:#4CAF50;">108.5 Hours</div>
                            </div>
                            <div>
                                <div style="font-size:0.75rem; color:#8B949E; text-transform:uppercase;">Averted Loss</div>
                                <div style="font-size:1.4rem; font-weight:800; color:#4CAF50;">$925,000</div>
                            </div>
                        </div>
                    </div>
                </div>
            """).strip(),
            unsafe_allow_html=True,
        )

        st.markdown("#### 📊 Comparative Hazard Mitigation Performance")
        modal_summary_table = """
| Depth Interval (MD) | Geohazard Event | Historical Failure in Offset Well (Volve-12B) | Active Well Mitigation Deployed (RIGX-TARGET-01) | Net NPT & Savings Realized | Operational Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **3,235 – 3,255 m** | Differential Stuck Pipe | Catastrophic pipe sticking in permeable sandstone; 48.0 hrs NPT; $420k jarring & acid treatment | Proactively limited rotary RPM to 60, dropped WOB to 14 klbs, spotted lubricating pill | **0 hrs NPT** (Saved 48.0 hrs; +$420,000) | <span style="color:#4CAF50; font-weight:bold;">✅ MITIGATED</span> |
| **3,280 – 3,300 m** | Severe Fractured Mud Loss | Total circulation loss (350 bbl/hr) upon penetrating fractured carbonate; 36.5 hrs NPT | Throttled mud pump to 450 GPM, spotted 75 bbl coarse LCM pill, boosted mud by +0.2 ppg | **Loss limited to 14 bbl/hr** (Saved 36.5 hrs; +$310,000) | <span style="color:#4CAF50; font-weight:bold;">✅ MITIGATED</span> |
| **3,330 – 3,350 m** | Borehole Pack-Off | Annular pack-off and bridge due to reactive shale sloughing; 24.0 hrs NPT | Performed bottoms-up circulation with high-density tandem sweep; 110 RPM agitation | **Full hole clearance maintained** (Saved 24.0 hrs; +$195,000) | <span style="color:#4CAF50; font-weight:bold;">✅ MITIGATED</span> |
"""
        st.markdown(modal_summary_table, unsafe_allow_html=True)

        if st.button("Close Summary", key="btn_close_eor_modal_p3", use_container_width=True):
            st.session_state["show_eor_modal"] = False
            st.rerun()

if st.session_state.get("show_eor_modal", False):
    render_end_of_run_dialog()

# Isolated Post-Run Container Ensuring Viewport Stability & Zero Column Shifting
with st.container():
    st.markdown("### 📋 Post-Run Well Evaluation & Benchmarking")

    # Check if final target depth is reached
    final_td = sim.FINAL_TARGET_DEPTH
    is_td_reached = sim.is_final_depth_reached(current_depth)

    with st.expander("End-of-Run Well Comparison Summary (Auto-Generated)", expanded=is_td_reached):
        if not is_td_reached:
            st.info(
                f"ℹ️ The End-of-Run comparison summary will automatically generate upon reaching final target depth "
                f"(Target TD: **{final_td:.1f} m MD**). Current Bit Depth: **{current_depth:.1f} m MD**."
            )
            col_td_btn, _ = st.columns([1.5, 2])
            with col_td_btn:
                if st.button("Simulate Reaching TD (3,365 m)", key="btn_sim_td", use_container_width=True):
                    sim.jump_to_final_row()
                    st.session_state["show_eor_modal"] = True
                    st.rerun()
        else:
            st.markdown(
                f"""
                <div style="background-color:#161B22; border:1px solid #4CAF50; border-radius:6px; padding:14px 18px; margin-bottom:16px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="background:rgba(76,175,80,0.2); color:#4CAF50; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:700;">
                                MISSION COMPLETED
                            </span>
                            <h4 style="color:#FFFFFF; margin:6px 0 2px 0;">Target Depth Reached: {current_depth:.1f} m MD</h4>
                            <div style="color:#8B949E; font-size:0.8rem;">Rig: RIGX-TARGET-01 • Subsurface Horizon Analysis Finalized</div>
                        </div>
                        <div style="display:flex; gap:20px; text-align:right;">
                            <div>
                                <div style="font-size:0.75rem; color:#8B949E;">TOTAL NPT SAVED</div>
                                <div style="font-size:1.2rem; font-weight:800; color:#4CAF50;">108.5 Hours</div>
                            </div>
                            <div>
                                <div style="font-size:0.75rem; color:#8B949E;">TOTAL COST AVERTED</div>
                                <div style="font-size:1.2rem; font-weight:800; color:#4CAF50;">$925,000</div>
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_modal_btn, _ = st.columns([1.5, 2])
            with col_modal_btn:
                if st.button("📋 Open End-of-Run Modal Window", key="btn_open_modal_p3", use_container_width=True):
                    st.session_state["show_eor_modal"] = True
                    st.rerun()

            st.markdown("#### 📊 Comparative Hazard Mitigation Performance")

            summary_table = """
| Depth Interval (MD) | Geohazard Event | Historical Failure in Offset Well (Volve-12B) | Active Well Mitigation Deployed (RIGX-TARGET-01) | Net NPT & Savings Realized | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **3,235 – 3,255 m** | Differential Stuck Pipe | Catastrophic pipe sticking in permeable sandstone; 48.0 hrs NPT; $420k jarring & acid treatment | Proactively limited rotary RPM to 60, dropped WOB to 14 klbs, spotted lubricating pill | **0 hrs NPT** (Saved 48.0 hrs; +$420,000) | <span style="color:#4CAF50; font-weight:bold;">✅ MITIGATED</span> |
| **3,280 – 3,300 m** | Severe Fractured Mud Loss | Total circulation loss (350 bbl/hr) upon penetrating fractured carbonate; 36.5 hrs NPT | Throttled mud pump to 450 GPM, spotted 75 bbl coarse LCM pill, boosted mud by +0.2 ppg | **Loss limited to 14 bbl/hr** (Saved 36.5 hrs; +$310,000) | <span style="color:#4CAF50; font-weight:bold;">✅ MITIGATED</span> |
| **3,330 – 3,350 m** | Borehole Pack-Off | Annular pack-off and bridge due to reactive shale sloughing; 24.0 hrs NPT | Performed bottoms-up circulation with high-density tandem sweep; 110 RPM agitation | **Full hole clearance maintained** (Saved 24.0 hrs; +$195,000) | <span style="color:#4CAF50; font-weight:bold;">✅ MITIGATED</span> |
"""
            st.markdown(summary_table, unsafe_allow_html=True)

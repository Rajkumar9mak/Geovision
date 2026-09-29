"""
GeoInsight-RigX (eRTMAC-NWIS)
Main Streamlit Application Entrypoint
Features: Native st.navigation & st.Page routing, Dark Industrial Design System,
and custom Interactive Glow effects.
"""

import streamlit as st

# Configure wide layout and page metadata
st.set_page_config(
    page_title="GeoInsight-RigX | eRTMAC-NWIS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from core_engine.telemetry_mocker import DrillingSimulator
from core_engine.ui_theme import inject_industrial_theme_css

# Inject Dark Industrial Theme & Glowing Button Interactions
inject_industrial_theme_css()

# Define native pages routing
pages = [
    st.Page("pages/1_🌍_Geospatial_Hub.py", title="Geospatial Hub", icon="🌍", default=True),
    st.Page("pages/2_📊_Log_Analytics.py", title="Log Analytics", icon="📊"),
    st.Page("pages/3_⚠️_Hazard_Advisories.py", title="Hazard Advisories", icon="⚠️"),
    st.Page("pages/4_🧠_Knowledge_Retrieval.py", title="Knowledge Retrieval", icon="🧠"),
]

pg = st.navigation(pages)

# Initialize persistent Drilling Simulator and active_rig_metrics in st.session_state
if "drilling_sim" not in st.session_state:
    st.session_state.drilling_sim = DrillingSimulator()
    st.session_state.drilling_sim.reset(start_offset=30)

sim = st.session_state.drilling_sim

if "active_rig_metrics" not in st.session_state:
    initial_hist = sim.get_streamed_history(limit=50)
    st.session_state.active_rig_metrics = {
        "current_depth": sim.get_current_depth(),
        "history_df": initial_hist,
        "latest_frame": initial_hist.iloc[-1].to_dict() if not initial_hist.empty else {},
        "is_streaming": True,
    }
else:
    # Progressively update active rig metrics on script re-executions to simulate live data ingestion
    if st.session_state.active_rig_metrics.get("is_streaming", True):
        next_frame = sim.step(n=1)
        st.session_state.active_rig_metrics["current_depth"] = next_frame["Depth"]
        st.session_state.active_rig_metrics["latest_frame"] = next_frame
        st.session_state.active_rig_metrics["history_df"] = sim.get_streamed_history(limit=250)

# Universal theme injection is executed via inject_industrial_theme_css()

# Native End-of-Run Modal Dialog Function
@st.dialog("📋 End-of-Run Well Comparison Summary (AAR)", width="large")
def render_end_of_run_modal():
    st.markdown(
        """
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
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 📊 Comparative Hazard Mitigation Performance")
    summary_table = """
| Depth Interval (MD) | Geohazard Event | Historical Failure in Offset Well (Volve-12B) | Active Well Mitigation Deployed (RIGX-TARGET-01) | Net NPT & Savings Realized | Operational Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **3,235 – 3,255 m** | Differential Stuck Pipe | Catastrophic pipe sticking in permeable sandstone; 48.0 hrs NPT; $420k jarring & acid treatment | Proactively limited rotary RPM to 60, dropped WOB to 14 klbs, spotted lubricating pill | **0 hrs NPT** (Saved 48.0 hrs; +$420,000) | <span style="color:#4CAF50; font-weight:bold;">✅ MITIGATED</span> |
| **3,280 – 3,300 m** | Severe Fractured Mud Loss | Total circulation loss (350 bbl/hr) upon penetrating fractured carbonate; 36.5 hrs NPT | Throttled mud pump to 450 GPM, spotted 75 bbl coarse LCM pill, boosted mud by +0.2 ppg | **Loss limited to 14 bbl/hr** (Saved 36.5 hrs; +$310,000) | <span style="color:#4CAF50; font-weight:bold;">✅ MITIGATED</span> |
| **3,330 – 3,350 m** | Borehole Pack-Off | Annular pack-off and bridge due to reactive shale sloughing; 24.0 hrs NPT | Performed bottoms-up circulation with high-density tandem sweep; 110 RPM agitation | **Full hole clearance maintained** (Saved 24.0 hrs; +$195,000) | <span style="color:#4CAF50; font-weight:bold;">✅ MITIGATED</span> |
"""
    st.markdown(summary_table, unsafe_allow_html=True)

    col_m_close, col_m_goto = st.columns([1, 1.2])
    with col_m_close:
        if st.button("Close Summary", key="btn_modal_close", use_container_width=True):
            st.session_state["show_eor_modal"] = False
            st.rerun()
    with col_m_goto:
        if st.button("🚨 View Mitigation Details", key="btn_modal_goto_hazard", use_container_width=True):
            st.session_state["show_eor_modal"] = False
            st.switch_page("pages/3_⚠️_Hazard_Advisories.py")

# Check if End-of-Run modal should be displayed
if st.session_state.get("show_eor_modal", False):
    render_end_of_run_modal()

# Sidebar Header & Metadata
with st.sidebar:
    st.markdown('<div class="rigx-status-badge">eRTMAC-NWIS</div>', unsafe_allow_html=True)
    st.markdown("### GeoInsight-RigX")
    st.caption("Real-Time Rig Monitoring & Subsurface Analytics")
    st.divider()

    # Active Live Telemetry Status Ticker
    active_metrics = st.session_state.get("active_rig_metrics", {})
    cur_d = active_metrics.get("current_depth", 3240.0)
    latest_dat = active_metrics.get("latest_frame", {})
    is_run_done = st.session_state.get("run_completed", False) or cur_d >= 3365.0
    status_label = "MISSION COMPLETED (TD)" if is_run_done else "ACTIVE DRILLING"
    status_color = "#4CAF50" if is_run_done else "#4CAF50"

    st.markdown("**⚡ Live Rig Status**")
    st.markdown(
        f"""
        <div style="background-color:#161B22; border:1px solid #30363D; border-radius:6px; padding:10px; font-size:0.8rem; margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#8B949E;">Bit Depth:</span>
                <b style="color:#E28743;">{cur_d:.1f} m MD</b>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#8B949E;">ROP:</span>
                <span style="color:#E6EDF3;">{latest_dat.get('ROP', 22.4):.1f} m/hr</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#8B949E;">WOB:</span>
                <span style="color:#E6EDF3;">{latest_dat.get('WOB', 25.0):.1f} klbs</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#8B949E;">Status:</span>
                <span style="color:{status_color}; font-weight:700;">{status_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if is_run_done:
        if st.button("📋 Open End-of-Run Modal", key="btn_sidebar_reopen_modal", use_container_width=True):
            st.session_state["show_eor_modal"] = True
            st.rerun()

    st.divider()

# ==============================================================================
# ENTERPRISE COMMAND HEADER & PERSISTENT GLOBAL ALERT OVERLAY
# ==============================================================================
st.markdown(
    """
    <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #30363D; padding-bottom:8px; margin-bottom:14px;">
        <div style="display:flex; align-items:center; gap:10px;">
            <span class="rigx-status-badge">eRTMAC-NWIS ENTERPRISE</span>
            <span style="font-size:1.05rem; font-weight:700; color:#F0F6FC;">GeoInsight-RigX Command Hub</span>
        </div>
        <div style="font-size:0.8rem; color:#8B949E;">
            Well: <b style="color:#E28743;">RIGX-TARGET-01</b> | System Mode: <span style="color:#4CAF50; font-weight:600;">● TELEMETRY STREAM ACTIVE</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Persistent UI Listener for Active Critical Warnings (Jitter-Free Unified Component)
is_critical = st.session_state.get("critical_warning", False)
active_hazard = st.session_state.get("active_hazard", None)

if is_critical and active_hazard is not None:
    hazard_title = active_hazard.get("title", "Imminent Geohazard Event")
    hazard_depth = active_hazard.get("depth_m", 0.0)
    dist_to_hazard = active_hazard.get("distance_to_event", 0.0)

    st.markdown('<div class="rigx-hazard-alert-banner">', unsafe_allow_html=True)
    with st.container():
        c_banner_msg, c_banner_btn = st.columns([3.8, 1.2], vertical_alignment="center")
        with c_banner_msg:
            st.markdown(
                f"""
                <div style="background-color: #8B0000; border: 2px solid #FF4444; border-radius: 8px; padding: 12px 18px; box-shadow: 0 0 16px rgba(255, 68, 68, 0.45); display: flex; align-items: center; gap: 14px;">
                    <span style="font-size: 1.8rem;">🚨</span>
                    <div>
                        <div style="color: #FFFFFF; font-weight: 800; font-size: 0.98rem; letter-spacing: 0.02em;">
                            CRITICAL WARNING: Imminent Hazard Zone Detected at Current Depth. Immediate Mitigation Required.
                        </div>
                        <div style="color: #FFCDD2; font-size: 0.82rem; margin-top: 2px;">
                            Offset Hazard: <b>{hazard_title}</b> at <b>{hazard_depth:.1f} m MD</b> (Proximity: within <b>{dist_to_hazard:.1f} m</b>).
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_banner_btn:
            if st.button("🚨 Enter Control Room", key="btn_global_mitigation", use_container_width=True):
                st.switch_page("pages/3_⚠️_Hazard_Advisories.py")
    st.markdown('</div>', unsafe_allow_html=True)

# Run the selected page
pg.run()


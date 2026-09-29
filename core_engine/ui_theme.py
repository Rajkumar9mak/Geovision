"""
GeoInsight-RigX (eRTMAC-NWIS)
Centralized UI Design System & Universal CSS Injection
Ensures dark industrial theme (#0E1117), glowing button interactions,
and strict Z-index / iframe isolation persist across all page switches.
"""

import streamlit as st

DARK_INDUSTRIAL_CSS = """
<style>
/* Dark Industrial Color Tokens */
:root {
    --bg-main: #0E1117;
    --bg-secondary: #161B22;
    --bg-card: #1F242D;
    --primary-color: #E28743;
    --primary-glow: rgba(226, 135, 67, 0.65);
    --text-primary: #E6EDF3;
    --text-muted: #8B949E;
    --border-color: #30363D;
    --hazard-red: #8B0000;
    --hazard-border: #FF4444;
}

/* Core App Background */
.stApp {
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
}

[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-color);
}

/* Universal Button Styling & Glowing Active States */
div.stButton > button,
div[data-testid="stFormSubmitButton"] > button,
div.stDialog button,
div[data-testid="stExpander"] button {
    background-color: #1F242D !important;
    color: #F0F6FC !important;
    border: 1px solid #30363D !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    padding: 0.55rem 1.25rem !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
}

/* Button Hover State */
div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover,
div.stDialog button:hover {
    border-color: #E28743 !important;
    color: #E28743 !important;
    background-color: #262C36 !important;
}

/* Button Active / Focus Glowing State */
div.stButton > button:active,
div.stButton > button:focus,
div[data-testid="stFormSubmitButton"] > button:active,
div[data-testid="stFormSubmitButton"] > button:focus,
div.stDialog button:active,
div.stDialog button:focus {
    border-color: #E28743 !important;
    outline: none !important;
    box-shadow: 0 0 14px 4px rgba(226, 135, 67, 0.7) !important;
    transform: scale(0.98) !important;
}

/* Metric Cards */
div[data-testid="stMetric"] {
    background-color: var(--bg-secondary);
    border: 1px solid var(--border-color);
    padding: 16px 20px;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
}
div[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
div[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* Status Badge */
.rigx-status-badge {
    display: inline-block;
    background: rgba(226, 135, 67, 0.15);
    border: 1px solid #E28743;
    color: #E28743;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

/* Z-INDEX & CONTAINER LAYERING ISOLATION */
/* 1. Global Hazard Alert Banner - Guaranteed Top Stacking Context */
.rigx-hazard-alert-banner {
    position: relative !important;
    z-index: 999999 !important;
}

/* 2. Strict Isolation for Folium / Leaflet / Plotly iframes */
iframe {
    border: none !important;
    background: transparent !important;
    position: relative !important;
    z-index: 1 !important;
}

.leaflet-container,
.leaflet-pane,
.leaflet-top,
.leaflet-bottom {
    z-index: 5 !important;
}

.stPlotlyChart {
    position: relative !important;
    z-index: 5 !important;
}

.js-plotly-plot .plotly .main-svg,
.user-select-none.svg-container {
    background: transparent !important;
}

/* 3. Streamlit Dialog Modal Overlays */
div[data-testid="stDialog"] {
    z-index: 1000000 !important;
}
div[data-testid="stDialog"] div[role="dialog"] {
    background-color: #161B22 !important;
    border: 1px solid #30363D !important;
    border-radius: 10px !important;
    color: #E6EDF3 !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.8) !important;
}
div[data-testid="stDialog"] header {
    background-color: #161B22 !important;
    color: #F0F6FC !important;
}

/* 4. Expander Smooth Expansion & Viewport Protection */
div[data-testid="stExpander"] {
    background-color: #161B22 !important;
    border: 1px solid #30363D !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    transition: all 0.3s ease-in-out !important;
    margin-top: 12px !important;
}
div[data-testid="stExpander"] summary {
    color: #F0F6FC !important;
    font-weight: 600 !important;
    padding: 10px 14px !important;
    background-color: #1F242D !important;
    border-radius: 6px !important;
}
div[data-testid="stExpander"] summary:hover {
    color: #E28743 !important;
}
div[data-testid="stExpander"] div[role="region"] {
    overflow-x: auto !important;
    max-width: 100% !important;
    padding: 12px 14px !important;
}
div[data-testid="stExpander"] table {
    width: 100% !important;
    border-collapse: collapse !important;
    margin: 8px 0 !important;
}
div[data-testid="stExpander"] th {
    background-color: #1F242D !important;
    color: #8B949E !important;
    padding: 8px 10px !important;
    font-size: 0.8rem !important;
    border: 1px solid #30363D !important;
    white-space: nowrap !important;
}
div[data-testid="stExpander"] td {
    background-color: #161B22 !important;
    color: #E6EDF3 !important;
    padding: 8px 10px !important;
    font-size: 0.82rem !important;
    border: 1px solid #30363D !important;
}
</style>
"""

def inject_industrial_theme_css():
    """Injects the dark industrial theme CSS universally into the current Streamlit render cycle."""
    st.markdown(DARK_INDUSTRIAL_CSS, unsafe_allow_html=True)

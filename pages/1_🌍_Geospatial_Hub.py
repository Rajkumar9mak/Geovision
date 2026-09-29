"""
GeoInsight-RigX (eRTMAC-NWIS)
Page 1: Geospatial Hub & Offset Well Intelligence
High-density dual-column interface for PostGIS offset scanning,
LAS/CSV ingestion QC, 2D proximity mapping, and 3D wellbore trajectory heatmaps.
"""

import math
import os
import textwrap
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Internal Core Engines
from core_engine.database_manager import SpatialDBManager
from core_engine.visualization_engine import generate_3d_heatmap, generate_sample_wellbore_data
from core_engine.log_parser import WellLogParser
from core_engine.ui_theme import inject_industrial_theme_css

# Ensure persistent dark industrial theme and Z-index isolation
inject_industrial_theme_css()

st.markdown("## 🌍 Geospatial Hub & Offset Well Intelligence")
st.caption("PostGIS Proximity Engine (ST_DWithin / ST_Distance) • Raw Log Ingestion • 3D Subsurface Trajectory Heatmap")

# Initialize database manager (cached in session state to maintain pool)
if "db_manager" not in st.session_state:
    st.session_state.db_manager = SpatialDBManager()

db_manager: SpatialDBManager = st.session_state.db_manager

# Initialize session state variables
if "target_lat" not in st.session_state:
    st.session_state.target_lat = 19.4215
if "target_lon" not in st.session_state:
    st.session_state.target_lon = 71.3510
if "radius_km" not in st.session_state:
    st.session_state.radius_km = 50.0
if "offset_wells" not in st.session_state:
    st.session_state.offset_wells = db_manager.get_nearest_offset_wells(
        st.session_state.target_lat,
        st.session_state.target_lon,
        radius_km=st.session_state.radius_km,
        limit=3,
    )
if "staged_well_log" not in st.session_state:
    st.session_state.staged_well_log = None
if "staged_metadata" not in st.session_state:
    st.session_state.staged_metadata = None


def update_selected_offset_wells(offset_df: pd.DataFrame, lat: float, lon: float, radius: float, selected_idx: int = 0):
    """
    Explicitly saves discovered offset wells, the chosen offset well identity,
    and its numerical baseline averages into st.session_state.selected_offset_wells
    for seamless cross-page data flow to Page 2 (Log Analytics).
    """
    if offset_df.empty:
        st.session_state.selected_offset_wells = None
        return None

    if selected_idx >= len(offset_df):
        selected_idx = 0

    primary = offset_df.iloc[selected_idx].to_dict()
    primary_name = str(primary.get("well_name", "OFFSET-WELL-01"))
    primary_td = max(float(primary.get("total_depth_m", 3500.0)), 3500.0)

    # Generate deterministic baseline curve dataset for the chosen offset well
    baseline_df = generate_sample_wellbore_data(well_name=primary_name, total_depth_m=primary_td, step_m=1.0)
    baseline_df = baseline_df[(baseline_df["DEPTH_M"] >= 3180.0) & (baseline_df["DEPTH_M"] <= 3450.0)].copy()
    baseline_df.rename(columns={"DEPTH_M": "Depth", "GR": "Gamma_Ray"}, inplace=True)
    if "RPM" not in baseline_df.columns:
        baseline_df["RPM"] = np.clip(115.0 + np.sin(baseline_df["Depth"] / 20.0) * 12.0, 90.0, 140.0)

    # Numerical baseline averages for the chosen well
    averages = {
        "avg_rop": round(float(baseline_df["ROP"].mean()), 1),
        "avg_wob": round(float(baseline_df["WOB"].mean()), 1),
        "avg_rpm": round(float(baseline_df["RPM"].mean()), 0),
        "avg_gr": round(float(baseline_df["Gamma_Ray"].mean()), 1),
    }

    state_obj = {
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
        "target_coordinates": {"lat": lat, "lon": lon, "radius_km": radius},
    }

    st.session_state.selected_offset_wells = state_obj
    return state_obj


# Initialize selected_offset_wells state if missing
if "selected_offset_wells" not in st.session_state or st.session_state.selected_offset_wells is None:
    update_selected_offset_wells(
        st.session_state.offset_wells,
        st.session_state.target_lat,
        st.session_state.target_lon,
        st.session_state.radius_km,
        selected_idx=0,
    )


# Helper to compute points along a spherical circle for radius visualization
def get_circle_coordinates(lat: float, lon: float, radius_km: float, num_points: int = 72):
    angles = np.linspace(0, 2 * np.pi, num_points)
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    # Earth radius in km
    r_earth = 6371.0
    angular_dist = radius_km / r_earth

    circle_lats = []
    circle_lons = []

    for a in angles:
        point_lat = math.asin(
            math.sin(lat_rad) * math.cos(angular_dist)
            + math.cos(lat_rad) * math.sin(angular_dist) * math.cos(a)
        )
        point_lon = lon_rad + math.atan2(
            math.sin(a) * math.sin(angular_dist) * math.cos(lat_rad),
            math.cos(angular_dist) - math.sin(lat_rad) * math.sin(point_lat),
        )
        circle_lats.append(math.degrees(point_lat))
        circle_lons.append(math.degrees(point_lon))

    return circle_lats, circle_lons


# Layout: Side-by-side high-density layout (1 : 2 ratio)
col_left, col_right = st.columns([1, 2], gap="medium")

# ==============================================================================
# LEFT COLUMN: CONTROL PANEL
# ==============================================================================
with col_left:
    st.markdown("### 🛠️ Control Panel")

    # DB Connection Indicator
    db_status_color = "#4CAF50" if db_manager.is_connected else "#FFA000"
    db_status_text = "PostGIS Active" if db_manager.is_connected else "Spatial Standalone"
    st.markdown(
        textwrap.dedent(f"""
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px; font-size:0.8rem; color:#8B949E;">
                <span style="height:9px; width:9px; background-color:{db_status_color}; border-radius:50%; display:inline-block;"></span>
                <span>Database Engine: <b style="color:#E6EDF3;">{db_status_text}</b></span>
            </div>
        """).strip(),
        unsafe_allow_html=True,
    )

    with st.form(key="spatial_scan_form"):
        st.markdown("**📍 Target Wellhead Location**")
        input_lat = st.number_input(
            "Target Latitude (°N)",
            value=float(st.session_state.target_lat),
            format="%.5f",
            step=0.005,
            help="WGS84 decimal latitude of the planned/active drilling rig.",
        )
        input_lon = st.number_input(
            "Target Longitude (°E)",
            value=float(st.session_state.target_lon),
            format="%.5f",
            step=0.005,
            help="WGS84 decimal longitude of the planned/active drilling rig.",
        )
        input_radius = st.slider(
            "Proximity Radius (km)",
            min_value=5.0,
            max_value=120.0,
            value=float(st.session_state.radius_km),
            step=5.0,
            help="PostGIS ST_DWithin radial threshold.",
        )

        scan_btn = st.form_submit_button("Scan Offset Wells", use_container_width=True)

        if scan_btn:
            st.session_state.target_lat = input_lat
            st.session_state.target_lon = input_lon
            st.session_state.radius_km = input_radius
            with st.spinner("Executing PostGIS spatial proximity query (ST_DWithin & ST_Distance)..."):
                st.session_state.offset_wells = db_manager.get_nearest_offset_wells(
                    input_lat, input_lon, radius_km=input_radius, limit=3
                )
                update_selected_offset_wells(st.session_state.offset_wells, input_lat, input_lon, input_radius)
            st.toast("Spatial offset scan complete! Baseline synced to Log Analytics.", icon="🎯")

    # Discovered Offset Wells Summary
    offset_df = st.session_state.offset_wells
    current_selected_idx = st.session_state.get("selected_offset_wells", {}).get("selected_index", 0) if st.session_state.get("selected_offset_wells") else 0

    if not offset_df.empty:
        st.markdown("#### 🎯 Identified Offset Wells", unsafe_allow_html=True)
        for idx, row in offset_df.iterrows():
            is_active = (idx == current_selected_idx)
            border_style = "2px solid #E28743" if is_active else "1px solid #30363D"
            badge_html = f'<span style="background:rgba(226,135,67,0.25); color:#E28743; padding:2px 6px; border-radius:3px; font-size:0.7rem; font-weight:700;">ACTIVE BASELINE</span>' if is_active else '<span></span>'

            offset_card_html = (
                f'<div style="border:{border_style}; background-color:#161B22; border-radius:6px; padding:10px 12px; margin-bottom:6px;">'
                f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                f'<b style="color:#F0F6FC; font-size:0.95rem;">{row["well_name"]}</b>'
                f'{badge_html}'
                f'</div>'
                f'<div style="font-size:0.78rem; color:#8B949E; margin-top:4px;">'
                f'<span>UWI: {row["uwi"]}</span> • <span>Operator: {row.get("operator", "N/A")}</span><br>'
                f'<span>Distance: <b style="color:#00E5FF;">{row["distance_km"]} km</b></span> • '
                f'<span>Total Well Depth: <b style="color:#E6EDF3;">{row.get("total_depth_m", "N/A")} m</b></span> • '
                f'<span>Status: <i style="color:#A5D6A7;">{row.get("status", "Active")}</i></span>'

                f'</div>'
                f'</div>'
            )
            st.markdown(offset_card_html, unsafe_allow_html=True)
            if not is_active:
                if st.button(f"🎯 Set as Active Baseline ({row['well_name']})", key=f"btn_set_baseline_{idx}", use_container_width=True):
                    update_selected_offset_wells(
                        offset_df,
                        st.session_state.target_lat,
                        st.session_state.target_lon,
                        st.session_state.radius_km,
                        selected_idx=idx,
                    )
                    st.toast(f"Switched baseline to {row['well_name']}! Log Analytics synced.", icon="🎯")
                    st.rerun()

    st.markdown("---")

    # Well Log Drag & Drop Ingestion Section
    st.markdown("#### 📥 Drag & Drop Well Log (.LAS / .CSV)")
    uploaded_file = st.file_uploader(
        "Upload raw log file for automated conditioning",
        type=["las", "csv"],
        help="Upload .LAS (v1.2, 2.0, 3.0) or .CSV well logs. Automatically drops nulls, standardizes depth to meters, and applies SciPy rolling median filter.",
    )

    if uploaded_file is not None:
        save_dir = Path("data_source/well_logs")
        save_dir.mkdir(parents=True, exist_ok=True)
        saved_path = save_dir / uploaded_file.name

        with open(saved_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process through WellLogParser
        with st.spinner(f"Ingesting & conditioning {uploaded_file.name}..."):
            try:
                parser = WellLogParser(data_dir=save_dir)
                cleaned_df, metadata = parser.parse_and_clean(
                    saved_path,
                    kernel_size=5,
                    target_depth_unit="M",
                    drop_strategy="any",
                )
                st.session_state.staged_well_log = cleaned_df
                st.session_state.staged_metadata = metadata

                st.success(f"Successfully processed: **{uploaded_file.name}**")
                st.markdown(
                    textwrap.dedent(f"""
                        <div style="background:#1F242D; border:1px solid #30363D; border-radius:6px; padding:10px; font-size:0.8rem; color:#E6EDF3;">
                            <div><b>Format:</b> {metadata.get('source_type')} | <b>Well:</b> {metadata.get('well_name')}</div>
                            <div><b>Rows:</b> {len(cleaned_df):,} | <b>Depth Standardized:</b> Meters (DEPTH_M)</div>
                            <div><b>Noise Filter:</b> SciPy rolling median (Kernel: 5)</div>
                        </div>
                    """).strip(),
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Error parsing log file: {e}")

# ==============================================================================
# RIGHT COLUMN: VISUALIZATION HUB
# ==============================================================================
with col_right:
    c_mhead, c_mengine = st.columns([1.7, 1.3], vertical_alignment="center")
    with c_mhead:
        st.markdown("### 🗺️ Geospatial Offset Map & 3D Horizon")
    with c_mengine:
        map_engine = st.radio(
            "Map Engine Selection",
            options=["Folium GIS (Isolated Sandboxed Frame)", "Plotly Spatial Engine"],
            horizontal=True,
            label_visibility="collapsed",
            help="Toggle between CSS-isolated Folium Leaflet Map and Plotly Vector Map."
        )

    target_lat = st.session_state.target_lat
    target_lon = st.session_state.target_lon
    radius_km = st.session_state.radius_km
    offset_df = st.session_state.offset_wells

    if "Folium" in map_engine:
        # 1A. Folium Map with Strict CSS & Z-Index Isolation via st.components.v1.html
        import folium

        m_folium = folium.Map(
            location=[target_lat, target_lon],
            zoom_start=9,
            tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
            attr="CartoDB Dark Matter",
            zoom_control=True,
        )

        # Proximity Buffer Circle
        folium.Circle(
            location=[target_lat, target_lon],
            radius=radius_km * 1000.0,
            color="#E28743",
            weight=2,
            fill=True,
            fill_color="#E28743",
            fill_opacity=0.08,
            popup=f"PostGIS Radial Buffer: {radius_km} km",
        ).add_to(m_folium)

        # Target Rig Wellhead Marker
        folium.Marker(
            location=[target_lat, target_lon],
            popup=f"<b>TARGET RIG (eRTMAC-NWIS)</b><br>Lat: {target_lat:.5f}°N<br>Lon: {target_lon:.5f}°E",
            tooltip="Target Rig Wellhead (Active)",
            icon=folium.DivIcon(
                html="""
                <div style="background-color:#E28743; width:16px; height:16px; border-radius:50%; border:2px solid #FFFFFF; box-shadow: 0 0 10px #E28743;"></div>
                """
            ),
        ).add_to(m_folium)

        # Discovered Offset Wells Markers
        if not offset_df.empty:
            for _, r in offset_df.iterrows():
                pop_content = (
                    f"<div style='font-family:sans-serif; font-size:12px; min-width:140px;'>"
                    f"<b style='color:#00E5FF; font-size:13px;'>{r['well_name']}</b><br>"
                    f"<span style='color:#8B949E;'>Distance:</span> <b style='color:#FFFFFF;'>{r['distance_km']} km</b><br>"
                    f"<span style='color:#8B949E;'>Operator:</span> {r.get('operator', 'N/A')}<br>"
                    f"<span style='color:#8B949E;'>Total Depth:</span> {r.get('total_depth_m', 'N/A')} m"
                    f"</div>"
                )
                folium.CircleMarker(
                    location=[r["latitude"], r["longitude"]],
                    radius=8,
                    color="#00E5FF",
                    weight=2,
                    fill=True,
                    fill_color="#00E5FF",
                    fill_opacity=0.9,
                    popup=pop_content,
                    tooltip=f"{r['well_name']} ({r['distance_km']} km away)",
                ).add_to(m_folium)

        # Isolated Sandboxed HTML Container (Zero Z-Index Bleed)
        folium_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                html, body {{ margin:0; padding:0; background: #0E1117; overflow: hidden; }}
                .leaflet-container {{ background: #0E1117 !important; border-radius: 8px; font-family: sans-serif; }}
                .leaflet-control-zoom {{ border: 1px solid #30363D !important; }}
                .leaflet-control-zoom a {{ background-color: #1F242D !important; color: #E6EDF3 !important; border-bottom: 1px solid #30363D !important; }}
                .leaflet-popup-content-wrapper {{ background: #161B22 !important; color: #E6EDF3 !important; border: 1px solid #30363D; border-radius: 6px; }}
                .leaflet-popup-tip {{ background: #161B22 !important; }}
            </style>
        </head>
        <body>
            <div style="position: relative; z-index: 1; border-radius: 8px; overflow: hidden; border: 1px solid #30363D;">
                {m_folium.get_root().render()}
            </div>
        </body>
        </html>
        """
        st.components.v1.html(folium_html, height=375)

    else:
        # 1B. 2D Interactive Proximity Map (Plotly Mapbox) with Transparent Backgrounds
        fig_map = go.Figure()
        ScatterMapClass = getattr(go, "Scattermap", getattr(go, "Scattermapbox", None))
        map_kwarg = "map" if hasattr(go, "Scattermap") else "mapbox"

        circ_lats, circ_lons = get_circle_coordinates(target_lat, target_lon, radius_km)
        fig_map.add_trace(
            ScatterMapClass(
                lat=circ_lats,
                lon=circ_lons,
                mode="lines",
                line=dict(color="#E28743", width=2),
                name=f"Radius: {radius_km} km",
                hoverinfo="name",
            )
        )

        if not offset_df.empty:
            hover_texts = [
                f"<b>{r['well_name']}</b><br>"
                f"Distance: {r['distance_km']} km<br>"
                f"Total Depth: {r['total_depth_m']} m<br>"
                f"Operator: {r.get('operator', 'N/A')}<br>"
                f"Status: {r.get('status', 'N/A')}"
                for _, r in offset_df.iterrows()
            ]

            fig_map.add_trace(
                ScatterMapClass(
                    lat=offset_df["latitude"],
                    lon=offset_df["longitude"],
                    mode="markers+text",
                    marker=dict(
                        size=14,
                        color="#00E5FF",
                        opacity=0.95,
                    ),
                    text=offset_df["well_name"],
                    textposition="top right",
                    textfont=dict(size=11, color="#F0F6FC"),
                    hovertext=hover_texts,
                    hoverinfo="text",
                    name="Discovered Offset Wells",
                )
            )

        fig_map.add_trace(
            ScatterMapClass(
                lat=[target_lat],
                lon=[target_lon],
                mode="markers+text",
                marker=dict(
                    size=18,
                    color="#E28743",
                    opacity=1.0,
                ),
                text=["TARGET RIG"],
                textposition="bottom right",
                textfont=dict(size=12, color="#E28743", family="monospace"),
                hovertext=[f"<b>Target Rig Wellhead</b><br>Lat: {target_lat:.5f}°N<br>Lon: {target_lon:.5f}°E"],
                hoverinfo="text",
                name="Target Rig (eRTMAC)",
            )
        )

        map_config = {
            "style": "carto-darkmatter",
            "center": dict(lat=target_lat, lon=target_lon),
            "zoom": 9.5,
        }
        fig_map.update_layout(
            **{map_kwarg: map_config},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=10),
            height=370,
            legend=dict(
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(14, 17, 23, 0.8)",
                bordercolor="#30363D",
                borderwidth=1,
                font=dict(color="#E6EDF3", size=10),
            ),
        )
        st.plotly_chart(fig_map, use_container_width=True)

    # 2. Interactive 3D Structural Heatmap for Primary Offset Well
    st.markdown("#### 🔬 Primary Offset Well: 3D Trajectory & Formation Heatmap")

    # Determine data source for 3D heatmap: Staged upload or Selected offset well baseline
    if st.session_state.staged_well_log is not None:
        source_name = f"Uploaded Log: {st.session_state.staged_metadata.get('well_name', 'Staged Well')}"
        heatmap_df = st.session_state.staged_well_log
    elif not offset_df.empty:
        curr_state = st.session_state.get("selected_offset_wells")
        primary_well = curr_state["primary_well"] if curr_state and "primary_well" in curr_state else offset_df.iloc[0]
        source_name = f"Active Baseline: {primary_well['well_name']} ({primary_well['distance_km']} km away)"
        total_d = float(primary_well.get("total_depth_m", 3500.0))
        heatmap_df = generate_sample_wellbore_data(well_name=primary_well["well_name"], total_depth_m=total_d)
    else:
        source_name = "Synthetic Offset Baseline"
        heatmap_df = generate_sample_wellbore_data(well_name="RIGX-BASELINE-01", total_depth_m=3400.0)

    # Parameter selection row for 3D visualizer
    c_source, c_metric, c_cmap = st.columns([2, 1, 1])

    with c_source:
        st.markdown(
            textwrap.dedent(f"""
                <div style="font-size:0.85rem; color:#8B949E; padding-top:6px;">
                    Visualizing Trajectory: <b style="color:#E28743;">{source_name}</b>
                </div>
            """).strip(),
            unsafe_allow_html=True,
        )

    # Candidate metrics from dataframe
    available_metrics = [
        c for c in heatmap_df.select_dtypes(include=[np.number]).columns
        if c.upper() not in ["DEPTH_M", "DEPT", "DEPTH", "X_EAST_M", "Y_NORTH_M", "X", "Y"]
    ]
    if not available_metrics:
        available_metrics = list(heatmap_df.columns)

    with c_metric:
        selected_metric = st.selectbox(
            "Formation Metric",
            options=available_metrics,
            index=0 if "GR" not in available_metrics else available_metrics.index("GR"),
            help="Petrophysical or MWD/LWD drilling metric mapped to 3D color scale.",
        )

    with c_cmap:
        selected_cmap = st.selectbox(
            "Colorscale",
            options=["Plasma", "Viridis", "Inferno", "Turbo", "Cividis"],
            index=0,
            help="High-density perceptual colormap.",
        )

    # Generate 3D Heatmap Figure
    fig_3d = generate_3d_heatmap(
        well_data_dataframe=heatmap_df,
        metric_col=selected_metric,
        colorscale=selected_cmap,
        title=f"{source_name} • {selected_metric} Horizon",
    )
    fig_3d.update_layout(
        height=480,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig_3d, use_container_width=True)

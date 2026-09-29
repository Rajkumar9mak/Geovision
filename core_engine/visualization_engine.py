"""
GeoInsight-RigX (eRTMAC-NWIS)
Core Engine - Visualization Engine
Provides 3D structural trajectory heatmaps and petrophysical visualizations
using Plotly Graph Objects integrated with the industrial dark theme.
"""

import math
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _detect_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Helper to detect a column matching candidate aliases case-insensitively."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for col, orig in lower_map.items():
        if any(cand.lower() in col for cand in candidates):
            return orig
    return None


def generate_sample_wellbore_data(
    well_name: str = "OFFSET-WELL-01",
    total_depth_m: float = 3400.0,
    step_m: float = 2.0,
) -> pd.DataFrame:
    """
    Generate realistic synthetic wellbore trajectory and telemetry logs
    (Depth, GR, WOB, ROP, Resistivity) for visual modeling and offset preview.
    Uses deterministic seeding from well_name to ensure each discovered offset well
    possesses distinct, reproducible geological and drilling baselines.
    """
    depths = np.arange(100.0, total_depth_m + step_m, step_m)
    n_points = len(depths)

    # Deterministic well seed for unique petrophysical characteristics
    seed = sum(ord(c) * (i + 1) for i, c in enumerate(str(well_name))) % 10000
    rng = np.random.RandomState(seed)
    depth_shift = (seed % 60) - 30.0
    rop_shift = ((seed % 15) - 7) * 0.85
    wob_shift = ((seed % 10) - 5) * 0.65
    gr_shift = ((seed % 20) - 10) * 1.4

    # Directional trajectory synthesis (Kickoff point at 800m, build and turn)
    x = np.zeros(n_points)
    y = np.zeros(n_points)
    tvd = np.zeros(n_points)
    tvd[0] = depths[0]

    kop = 800.0
    azimuth_base = 35.0 + (seed % 40)
    build_rate = 50.0 + (seed % 15)
    max_inclination = 17.5 + (seed % 5) * 0.6
    for i, d in enumerate(depths):
        if d > kop:
            delta_d = d - kop
            inclination_deg = min(max_inclination, (delta_d / 1200.0) * build_rate)
            azimuth_deg = azimuth_base + (delta_d / 1800.0) * 20.0
            inc_rad = math.radians(inclination_deg)
            az_rad = math.radians(azimuth_deg)

            # Cumulative displacement
            x[i] = x[i - 1] + (step_m * math.sin(inc_rad) * math.sin(az_rad))
            y[i] = y[i - 1] + (step_m * math.sin(inc_rad) * math.cos(az_rad))
            tvd[i] = tvd[i - 1] + (step_m * math.cos(inc_rad))
        else:
            tvd[i] = d

    # Synthetic Petrophysical & Drilling Metrics with well-specific formation shifts
    base_strata = np.sin((depths + depth_shift) / 115.0) * 32.0 + np.cos(depths / 42.0) * 16.0
    noise_gr = rng.normal(0, 3.0, n_points)
    gr = np.clip(72.0 + gr_shift + base_strata + noise_gr, 15.0, 160.0)

    # WOB (klbs: 10 to 45)
    wob = np.clip(25.0 + wob_shift + np.sin((depths + depth_shift) / 85.0) * 8.5 + rng.normal(0, 1.8, n_points), 8.0, 50.0)

    # ROP (m/hr: 5 to 35)
    rop = np.clip(20.5 + rop_shift - (wob - 25.0) * 0.42 + rng.normal(0, 1.5, n_points), 3.0, 42.0)

    # Resistivity (ohm.m: 0.5 to 150 log scale)
    res = np.clip(np.exp((gr - 50.0) / 32.0) + rng.normal(0, 0.8, n_points), 0.2, 200.0)

    # Hydrostatic & baseline formation pressure profile (psi)
    base_pressure = 4250.0 + (tvd - 3168.0) * 1.35 + (wob_shift * 2.0)
    pressure = np.clip(base_pressure + rng.normal(0, 1.2, n_points), 3600.0, 5200.0)

    df = pd.DataFrame({
        "DEPTH_M": depths,
        "X_EAST_M": np.round(x, 2),
        "Y_NORTH_M": np.round(y, 2),
        "GR": np.round(gr, 2),
        "WOB": np.round(wob, 2),
        "ROP": np.round(rop, 2),
        "RES": np.round(res, 2),
        "TVD": np.round(tvd, 2),
        "Pressure": np.round(pressure, 1),
    })
    return df


def generate_3d_heatmap(
    well_data_dataframe: pd.DataFrame,
    metric_col: Optional[str] = None,
    depth_col: Optional[str] = None,
    colorscale: str = "Plasma",
    title: Optional[str] = None,
) -> go.Figure:
    """
    Generate an interactive 3D wellbore trajectory heatmap using Plotly Graph Objects.
    
    Plots depth on the Z-axis, with X and Y representing the spatial spread of the wellbore,
    mapping a key drilling/logging metric (like Gamma Ray or WOB) to a high-density, interactive 3D colorscale.
    
    Paper and plot backgrounds are explicitly set to transparent ('rgba(0,0,0,0)')
    to seamlessly blend into GeoInsight-RigX's dark industrial design system.
    
    Parameters:
        well_data_dataframe: DataFrame containing depth and measurement curves.
        metric_col: Column name to map to colorscale (e.g. 'GR', 'WOB', 'ROP'). Auto-detected if None.
        depth_col: Column name representing Depth (meters). Auto-detected if None.
        colorscale: Plotly colorscale ('Plasma', 'Viridis', 'Inferno', 'Turbo', etc.).
        title: Optional title string.
    
    Returns:
        go.Figure: Ready-to-render Plotly 3D Figure.
    """
    df = well_data_dataframe.copy()
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    # 1. Resolve Depth Column (Z-Axis)
    detected_depth = depth_col or _detect_column(df, ["DEPTH_M", "DEPT", "DEPTH", "MD", "TVD"])
    if detected_depth is None:
        # Fall back to index or first numeric column
        z_vals = df.index.to_numpy(dtype=float)
        depth_label = "Measured Depth (m)"
    else:
        z_vals = df[detected_depth].to_numpy(dtype=float)
        depth_label = f"Depth: {detected_depth} (m)"

    n_points = len(z_vals)

    # 2. Resolve or Synthesize Spatial Spread (X and Y Axes)
    detected_x = _detect_column(df, ["X_EAST_M", "EASTING", "X", "DX", "DISPLACEMENT_X"])
    detected_y = _detect_column(df, ["Y_NORTH_M", "NORTHING", "Y", "DY", "DISPLACEMENT_Y"])

    if detected_x and detected_y:
        x_vals = df[detected_x].to_numpy(dtype=float)
        y_vals = df[detected_y].to_numpy(dtype=float)
    else:
        # Synthesize realistic directional well trajectory spread from depth
        # Kickoff at 25% depth with gradual build and dogleg turn
        kop = z_vals[0] + (z_vals[-1] - z_vals[0]) * 0.22
        x_vals = np.zeros(n_points)
        y_vals = np.zeros(n_points)

        for i in range(1, n_points):
            d = z_vals[i]
            if d > kop:
                excess = d - kop
                inc = math.radians(min(45.0, (excess / max(100.0, (z_vals[-1] - kop))) * 45.0))
                az = math.radians(40.0 + (excess / max(100.0, (z_vals[-1] - kop))) * 25.0)
                step = abs(z_vals[i] - z_vals[i - 1])
                x_vals[i] = x_vals[i - 1] + (step * math.sin(inc) * math.sin(az))
                y_vals[i] = y_vals[i - 1] + (step * math.sin(inc) * math.cos(az))
            else:
                x_vals[i] = 0.0
                y_vals[i] = 0.0

    # 3. Resolve Metric for High-Density Colorscale
    if metric_col and metric_col in df.columns:
        chosen_metric = metric_col
    else:
        # Prefer Gamma Ray, WOB, ROP, Resistivity
        preferred_metrics = ["GR", "GAMMA", "WOB", "ROP", "RES", "RT", "RHOB", "NPHI", "TORQUE"]
        found = _detect_column(df, preferred_metrics)
        if found:
            chosen_metric = found
        else:
            # Pick first numeric column that is not depth/spatial
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            filtered_cols = [c for c in numeric_cols if c not in [detected_depth, detected_x, detected_y]]
            chosen_metric = filtered_cols[0] if filtered_cols else detected_depth

    metric_values = df[chosen_metric].to_numpy(dtype=float)

    # 4. Construct High-Density 3D Scatter & Line Trace
    hover_texts = [
        f"<b>Depth:</b> {z:.1f} m<br>"
        f"<b>{chosen_metric}:</b> {val:.2f}<br>"
        f"<b>East/West (X):</b> {x:.1f} m<br>"
        f"<b>North/South (Y):</b> {y:.1f} m"
        for z, val, x, y in zip(z_vals, metric_values, x_vals, y_vals)
    ]

    # Primary 3D Trajectory Trace
    trace_trajectory = go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode="lines+markers",
        line=dict(
            color=metric_values,
            colorscale=colorscale,
            width=7,
        ),
        marker=dict(
            size=4.5,
            color=metric_values,
            colorscale=colorscale,
            opacity=0.92,
            showscale=True,
            colorbar=dict(
                title=dict(
                    text=f"{chosen_metric}",
                    font=dict(color="#E6EDF3", size=13, family="sans-serif"),
                    side="right",
                ),
                tickfont=dict(color="#8B949E", size=11),
                thickness=16,
                len=0.75,
                bgcolor="rgba(22, 27, 34, 0.8)",
                bordercolor="#30363D",
                borderwidth=1,
            ),
        ),
        text=hover_texts,
        hoverinfo="text",
        name=f"Wellbore ({chosen_metric})",
    )

    # Surface Rig Wellhead Marker (0, 0, Surface)
    surface_wellhead = go.Scatter3d(
        x=[x_vals[0]],
        y=[y_vals[0]],
        z=[z_vals[0]],
        mode="markers+text",
        marker=dict(
            size=9,
            color="#E28743",  # Primary amber
            symbol="diamond",
            line=dict(color="#FFFFFF", width=1.5),
        ),
        text=["Surface Wellhead"],
        textposition="top center",
        textfont=dict(color="#E28743", size=12),
        hoverinfo="text",
        name="Wellhead",
    )

    # Bottom Hole Assembly (BHA) Target Marker
    bha_marker = go.Scatter3d(
        x=[x_vals[-1]],
        y=[y_vals[-1]],
        z=[z_vals[-1]],
        mode="markers+text",
        marker=dict(
            size=8,
            color="#00E5FF",  # Cyan highlight
            symbol="circle",
            line=dict(color="#FFFFFF", width=1.5),
        ),
        text=["BHA Total Depth"],
        textposition="bottom center",
        textfont=dict(color="#00E5FF", size=11),
        hoverinfo="text",
        name="TD / Bit",
    )

    fig = go.Figure(data=[trace_trajectory, surface_wellhead, bha_marker])

    # 5. Industrial Dark Theme Layout & Transparent Backgrounds
    fig.update_layout(
        title=dict(
            text=title or f"3D Structural Trajectory & Heatmap ({chosen_metric})",
            font=dict(color="#F0F6FC", size=15),
            x=0.03,
            y=0.96,
        ),
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
        plot_bgcolor="rgba(0,0,0,0)",   # Transparent background
        autosize=True,
        margin=dict(l=10, r=10, b=15, t=40),
        legend=dict(
            font=dict(color="#E6EDF3", size=11),
            bgcolor="rgba(14, 17, 23, 0.7)",
            bordercolor="#30363D",
            borderwidth=1,
            x=0.02,
            y=0.95,
        ),
        scene=dict(
            bgcolor="rgba(0,0,0,0)",  # Transparent 3D scene
            camera=dict(
                eye=dict(x=1.65, y=1.65, z=1.2),
                up=dict(x=0, y=0, z=1),
            ),
            xaxis=dict(
                title=dict(text="East-West Displacement (m)", font=dict(color="#8B949E", size=11)),
                backgroundcolor="rgba(22, 27, 34, 0.5)",
                gridcolor="#30363D",
                showbackground=True,
                zerolinecolor="#E28743",
                tickfont=dict(color="#8B949E", size=10),
            ),
            yaxis=dict(
                title=dict(text="North-South Displacement (m)", font=dict(color="#8B949E", size=11)),
                backgroundcolor="rgba(22, 27, 34, 0.5)",
                gridcolor="#30363D",
                showbackground=True,
                zerolinecolor="#E28743",
                tickfont=dict(color="#8B949E", size=10),
            ),
            zaxis=dict(
                title=dict(text=depth_label, font=dict(color="#8B949E", size=11)),
                backgroundcolor="rgba(22, 27, 34, 0.5)",
                gridcolor="#30363D",
                showbackground=True,
                zerolinecolor="#30363D",
                tickfont=dict(color="#8B949E", size=10),
                autorange="reversed",  # Depth increases downwards into earth
            ),
        ),
    )

    return fig

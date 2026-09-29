"""
Generate synthetic active well telemetry time-series dataset.
"""
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

def generate_telemetry_csv(output_path: str = "data_source/production/active_rig_telemetry.csv", n_records: int = 600):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    start_time = datetime.datetime.now() - datetime.timedelta(seconds=n_records)
    timestamps = [start_time + datetime.timedelta(seconds=i) for i in range(n_records)]
    
    start_depth = 3200.0
    final_depth = 3365.0

    base_depths = np.linspace(start_depth, final_depth, n_records)
    noise_d = np.cumsum(np.random.normal(0, 0.02, n_records))
    noise_d -= np.linspace(noise_d[0], noise_d[-1], n_records)
    depths = np.round(base_depths + noise_d, 2)
    depths[-1] = 3365.0
        
    # ROP with hazard intervals
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

    # WOB between 18 and 36 klbs
    wob = np.clip(26.0 + np.sin(depths / 15.0) * 6.0 + np.random.normal(0, 1.5, n_records), 14.0, 42.0)
    
    # RPM between 95 and 150
    rpm = np.clip(120.0 + np.cos(depths / 20.0) * 15.0 + np.random.normal(0, 3.0, n_records), 80.0, 160.0)
    
    # Gamma Ray (API) with geological bed boundaries
    lithology_strata = np.sin(depths / 25.0) * 35.0 + np.cos(depths / 8.0) * 12.0
    gr = np.clip(75.0 + lithology_strata + np.random.normal(0, 2.5, n_records), 20.0, 145.0)
    
    # Deviated inclination & TVD
    inc = np.clip(17.5 + (depths - 3200.0) * 0.010 + 0.12 * np.sin(depths / 14.0), 15.0, 25.0)
    tvd = np.zeros(n_records)
    tvd[0] = 3168.0 + (depths[0] - 3200.0) * np.cos(np.radians(inc[0]))
    for i in range(1, n_records):
        d_md = depths[i] - depths[i - 1]
        tvd[i] = tvd[i - 1] + d_md * np.cos(np.radians(inc[i]))

    # Realistic drilling pressure (psi)
    p_hydro = 4268.0 + 1.35 * (tvd - 3168.0)
    p_circ = 0.35 * (wob - 25.0) + 0.22 * (rop - 20.0) + 0.12 * (rpm - 120.0)
    p_fluct = 3.5 * np.sin(depths / 16.0) + 1.8 * np.cos(depths / 7.0)
    h_stuck = 55.0 * np.exp(-((depths - 3245.0) / 2.8) ** 2)
    h_loss = -220.0 * np.exp(-((depths - 3290.0) / 2.8) ** 2)
    h_pack = 310.0 * np.exp(-((depths - 3340.0) / 2.8) ** 2)
    pressure = np.clip(p_hydro + p_circ + p_fluct + h_stuck + h_loss + h_pack, 3600.0, 5200.0)

    df = pd.DataFrame({
        "Timestamp": [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps],
        "Depth": np.round(depths, 2),
        "TVD": np.round(tvd, 2),
        "Inclination": np.round(inc, 2),
        "ROP": np.round(rop, 2),
        "WOB": np.round(wob, 2),
        "RPM": np.round(rpm, 1),
        "Gamma_Ray": np.round(gr, 2),
        "Pressure": np.round(pressure, 1),
    })
    
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} telemetry records to {output_path}")

if __name__ == "__main__":
    generate_telemetry_csv()

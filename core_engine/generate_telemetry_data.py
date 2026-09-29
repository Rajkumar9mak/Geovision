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
    depths = np.zeros(n_records)
    current_d = start_depth
    
    # ROP between 12 and 34 m/hr
    base_rop = 20.0 + np.sin(np.linspace(0, 15, n_records)) * 8.0 + np.random.normal(0, 1.2, n_records)
    rop = np.clip(base_rop, 8.0, 38.0)
    
    # Depth increment per second: (ROP m/hr) / 3600 * acceleration_factor (e.g. 5x for real-time visualization delight)
    for i in range(n_records):
        current_d += (rop[i] / 3600.0) * 12.0  # ~0.06m per second
        depths[i] = round(current_d, 2)
        
    # WOB between 18 and 36 klbs
    wob = np.clip(26.0 + np.sin(depths / 15.0) * 6.0 + np.random.normal(0, 1.5, n_records), 14.0, 42.0)
    
    # RPM between 95 and 150
    rpm = np.clip(120.0 + np.cos(depths / 20.0) * 15.0 + np.random.normal(0, 3.0, n_records), 80.0, 160.0)
    
    # Gamma Ray (API) with geological bed boundaries
    lithology_strata = np.sin(depths / 25.0) * 35.0 + np.cos(depths / 8.0) * 12.0
    gr = np.clip(75.0 + lithology_strata + np.random.normal(0, 2.5, n_records), 20.0, 145.0)
    
    df = pd.DataFrame({
        "Timestamp": [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps],
        "Depth": np.round(depths, 2),
        "ROP": np.round(rop, 2),
        "WOB": np.round(wob, 2),
        "RPM": np.round(rpm, 1),
        "Gamma_Ray": np.round(gr, 2),
    })
    
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} telemetry records to {output_path}")

if __name__ == "__main__":
    generate_telemetry_csv()

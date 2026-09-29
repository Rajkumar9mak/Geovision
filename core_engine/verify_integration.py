"""
Integration Verification Test for GeoInsight-RigX (eRTMAC-NWIS)
Tests:
1. SpatialDBManager proximity search (PostGIS / Haversine)
2. Deterministic baseline curves & numerical averages per well
3. DrillingSimulator telemetry stream, hazard detection, and clean TD halt
4. SubsurfaceRAGPipeline query retrieval
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

from core_engine.database_manager import SpatialDBManager
from core_engine.visualization_engine import generate_sample_wellbore_data
from core_engine.telemetry_mocker import DrillingSimulator
from core_engine.rag_pipeline import query_historical_reports

def test_integration():
    print("=== TEST 1: PostGIS Proximity Search ===")
    db = SpatialDBManager()
    wells = db.get_nearest_offset_wells(19.4215, 71.3510, radius_km=50.0, limit=3)
    assert len(wells) == 3, f"Expected 3 wells, got {len(wells)}"
    for idx, r in wells.iterrows():
        print(f"  Well {idx + 1}: {r['well_name']} | Distance: {r['distance_km']} km | TD: {r['total_depth_m']} m")

    print("\n=== TEST 2: Deterministic Baseline Curves & Averages ===")
    averages = {}
    for idx, r in wells.iterrows():
        w_name = r["well_name"]
        w_td = max(float(r.get("total_depth_m", 3500.0)), 3500.0)
        b_df = generate_sample_wellbore_data(well_name=w_name, total_depth_m=w_td, step_m=1.0)
        interval = b_df[(b_df["DEPTH_M"] >= 3180.0) & (b_df["DEPTH_M"] <= 3450.0)]
        avg_rop = round(float(interval["ROP"].mean()), 1)
        avg_wob = round(float(interval["WOB"].mean()), 1)
        avg_gr = round(float(interval["GR"].mean()), 1)
        averages[w_name] = (avg_rop, avg_wob, avg_gr)
        print(f"  {w_name}: avg ROP = {avg_rop} m/hr, avg WOB = {avg_wob} klbs, avg GR = {avg_gr} API")

    # Verify that different wells have different baseline averages
    names = list(averages.keys())
    assert averages[names[0]] != averages[names[1]], "Wells should have distinct baseline characteristics"
    print("  [OK] Wells have proven distinct baseline characteristics!")

    print("\n=== TEST 3: Telemetry Stream Progression & Final Row TD Halt ===")
    sim = DrillingSimulator()
    start_d = sim.get_current_depth()
    print(f"  Initial Depth: {start_d:.1f} m MD")

    # Test hazard check
    h1 = sim.check_hazard_proximity(3245.0)
    assert h1 is not None and h1["hazard_type"] == "Stuck Pipe", "Should detect Stuck Pipe at 3245m"
    print(f"  [OK] Hazard proximity detected: {h1['title']} at {h1['depth_m']} m")

    h2 = sim.check_hazard_proximity(3290.0)
    assert h2 is not None and h2["hazard_type"] == "Mud Loss", "Should detect Mud Loss at 3290m"
    print(f"  [OK] Hazard proximity detected: {h2['title']} at {h2['depth_m']} m")

    # Test jump to final row
    final_point = sim.jump_to_final_row()
    print(f"  Final Row Depth: {final_point['Depth']:.1f} m MD")
    assert final_point["Depth"] == 3365.0, f"Expected 3365.0, got {final_point['Depth']}"
    assert sim.is_streaming is False, "Simulator should halt streaming at final row"
    assert sim.is_run_completed is True, "Simulator should mark run completed"
    print("  [OK] Simulator cleanly halted and marked run completed at CSV final row!")

    print("\n=== TEST 4: Open-Source RAG Knowledge Retrieval ===")
    results = query_historical_reports("stuck pipe sandstone Heimdal", k=2)
    print(f"  Retrieved {len(results)} chunks from ChromaDB:")
    for r in results:
        print(f"    - [{r['source_file']} p.{r['page_number']}] {r['content'][:90]}...")
    assert len(results) > 0, "Should retrieve RAG chunks"
    print("  [OK] RAG pipeline operational!")

    print("\nALL INTEGRATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_integration()

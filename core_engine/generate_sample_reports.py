"""
Utility to generate realistic PDF reports for RAG pipeline ingestion.
Creates valid PDF-1.4 documents containing geological hazard case studies,
drilling failure analyses, and mitigation guidelines.
"""

from pathlib import Path
from typing import List, Dict

def create_pdf(filename: Path, title: str, pages_content: List[Dict[str, str]]):
    """
    Generate a valid multi-page PDF document without third-party heavy tools.
    """
    filename.parent.mkdir(parents=True, exist_ok=True)
    
    objects = []
    
    def add_object(content: str) -> int:
        objects.append(content)
        return len(objects)

    # We will build objects list
    # Object 1: Catalog (will point to Pages)
    # Object 2: Pages (kids will be page objects)
    # Object 3: Font
    font_id = 3
    
    page_ids = []
    stream_ids = []
    
    # Pre-reserve object slots
    objects.append("") # 1: Catalog
    objects.append("") # 2: Pages
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>") # 3: Font
    
    for i, p_info in enumerate(pages_content):
        # Create Content Stream
        text_lines = []
        text_lines.append(f"BT /F1 16 Tf 50 750 Td ({title}) Tj ET")
        text_lines.append(f"BT /F1 12 Tf 50 725 Td (Section: {p_info.get('section', 'General')}) Tj ET")
        text_lines.append(f"BT /F1 10 Tf 50 705 Td (Page {i+1} of {len(pages_content)} - GeoInsight-RigX Historical Archives) Tj ET")
        
        y = 670
        for para in p_info.get("paragraphs", []):
            words = para.split(" ")
            line = ""
            for w in words:
                if len(line) + len(w) + 1 > 75:
                    text_lines.append(f"BT /F1 10 Tf 50 {y} Td ({line.strip()}) Tj ET")
                    y -= 14
                    line = w + " "
                else:
                    line += w + " "
            if line.strip():
                text_lines.append(f"BT /F1 10 Tf 50 {y} Td ({line.strip()}) Tj ET")
                y -= 22
                
        stream_content = "\n".join(text_lines)
        stream_obj = f"<< /Length {len(stream_content.encode('utf-8'))} >>\nstream\n{stream_content}\nendstream"
        objects.append(stream_obj)
        stream_id = len(objects)
        stream_ids.append(stream_id)
        
        page_obj = f"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 3 0 R >> >> /MediaBox [0 0 612 792] /Contents {stream_id} 0 R >>"
        objects.append(page_obj)
        page_id = len(objects)
        page_ids.append(page_id)

    # Now populate Catalog and Pages
    objects[0] = "<< /Type /Catalog /Pages 2 0 R >>"
    kids_str = " ".join([f"{pid} 0 R" for pid in page_ids])
    objects[1] = f"<< /Type /Pages /Kids [{kids_str}] /Count {len(page_ids)} >>"
    
    # Write PDF file with xref table
    with open(filename, "wb") as f:
        f.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = []
        for i, obj in enumerate(objects):
            offsets.append(f.tell())
            f.write(f"{i+1} 0 obj\n{obj}\nendobj\n".encode("utf-8"))
            
        xref_pos = f.tell()
        f.write(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode("utf-8"))
        for offset in offsets:
            f.write(f"{offset:010d} 00000 n \n".encode("utf-8"))
            
        f.write(f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("utf-8"))


def generate_all_reports(reports_dir: str = "data_source/reports"):
    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Report 1: Volve Well 15/9-F-12B End of Well Geological & Hazards Report
    p1 = [
        {
            "section": "Stratigraphy & Overview",
            "paragraphs": [
                "Well 15/9-F-12B was spudded as a production well targeting the Hugin and Heimdal sandstone reservoir intervals. The primary geomechanical objective was reaching 3450m MD through complex interbedded shale and permeable sands.",
                "Subsurface stratigraphic horizons include Hordaland shale from 1800m to 2900m, Balder reactive smectite formation at 3100m, and Heimdal sandstone formation between 3230m and 3270m MD."
            ]
        },
        {
            "section": "Extraction Failure Notes & Sandstone Geohazards",
            "paragraphs": [
                "Extraction failure notes in the adjacent sandstone layers indicate severe differential sticking at 3245m MD. The active BHA became immobilized after a 15-minute connection due to 450 psi differential overbalance.",
                "Formation evaluation revealed highly permeable sandstone (porosity 28%, permeability 850 mD). High mud cake thickness and high hydrostatic overbalance caused mechanical binding of the drill collars against the borehole wall.",
                "Immediate remediation required continuous jarring for 36 hours and pumping 50 bbl of pipe-freeing lubricating chemical wash. Total non-productive time (NPT) was 48.0 hours with remediation cost exceeding $420,000.",
                "Mitigation recommendations for future offset drilling: Maintain rotary speed at 60 RPM, limit stationary time to under 3 minutes, maintain WOB below 14 klbs, and optimize fluid loss additives to ensure a ultra-thin, low-friction mud filter cake."
            ]
        },
        {
            "section": "Carbonate Fractures & Severe Mud Losses",
            "paragraphs": [
                "At 3290m MD within the Lower Hordaland fractured carbonate boundary, sudden partial to total mud losses occurred at a rate exceeding 350 bbl/hr.",
                "Hydrostatic head decreased by 180 psi, prompting immediate deployment of 75 bbl coarse and medium LCM pills. Pump flow rate was throttled to 450 GPM to reduce Equivalent Circulating Density (ECD) until losses subsided.",
                "Recommended mitigation: Increase mud weight by 0.2 ppg incrementally once fractures seal to ensure borehole mechanical stability while monitoring PWD sensors."
            ]
        }
    ]
    create_pdf(out_dir / "Volve_15_9_F12B_Geological_Hazards_Report.pdf", "Volve 15/9-F-12B Geological & Drilling Hazards Report", p1)
    
    # Report 2: Gulf Offshore Stuck Pipe Prevention & Mechanical Pack-off Study
    p2 = [
        {
            "section": "Mechanics of Mechanical Binding and Pack-Off",
            "paragraphs": [
                "Mechanical sticking and pack-off hazards in extended-reach drilling are predominantly driven by inefficient hole cleaning, cuttings bed accumulation, and reactive shale swelling.",
                "In deep intervals (>3000m MD), inadequate annular velocity allows cuttings to build up on the low side of the hole. When pulling out of hole or rotating below 90 RPM, the BHA acts as a wedge, precipitating instantaneous pack-off and pressure spikes exceeding 400 psi."
            ]
        },
        {
            "section": "Proactive Mitigation & Hydraulic Hole Cleaning Directives",
            "paragraphs": [
                "Drilling crews in reactive smectite shale must deploy tandem high-density viscous pills every 90 meters drilled.",
                "Maintain drillstring rotation at 110-130 RPM during reaming and reciprocate pipe smoothly to agitate cuttings beds into high-velocity laminar flow channels.",
                "If pump pressure surges unexpectedly with torque fluctuations, stop penetration immediately, pull bit off bottom by 3 to 5 meters, and circulate bottoms-up at maximum allowable flow rate before resuming drilling."
            ]
        }
    ]
    create_pdf(out_dir / "Offshore_Stuck_Pipe_Mitigation_Best_Practices.pdf", "Offshore Stuck Pipe & Pack-off Prevention Study", p2)
    
    # Report 3: High-Pressure Formation Influx & Wellbore Stability Guidelines
    p3 = [
        {
            "section": "Pore Pressure & Hydrostatic Balance",
            "paragraphs": [
                "Accurate pore pressure prediction and real-time kick detection are paramount during development drilling across depleted sandstone reservoirs.",
                "Differential pressure across the drill bit must not exceed 350 psi in high-porosity intervals to mitigate differential sticking risk.",
                "Continuous monitoring of pit volume totals, flow paddles, and mud temperature gradients provides early warning of impending kicks or fracture-induced fluid loss."
            ]
        }
    ]
    create_pdf(out_dir / "HPHT_Drilling_Hazards_And_Wellbore_Stability.pdf", "HPHT Wellbore Stability & Influx Management Guidelines", p3)
    
    print(f"Successfully generated 3 technical PDF reports in {out_dir}")

if __name__ == "__main__":
    generate_all_reports()

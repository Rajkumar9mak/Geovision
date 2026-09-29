"""
GeoInsight-RigX (eRTMAC-NWIS)
Module: Well Extractor Engine (Strict Current-Document Grounding)
Extracts factual well metadata, engineering parameters, categorized valuable information,
and concise report extracts directly from the current document.
STRICT RULE: NEVER invent values. If information is not in the text, returns "Not found in report".
"""

import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("WellExtractor")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class WellExtractorEngine:
    """
    Factual information extractor strictly grounded in the currently uploaded document.
    """

    @classmethod
    def extract_well_information(cls, pages_data: List[Dict[str, Any]], filename: str) -> Dict[str, Any]:
        """
        Extract factual well information only if present in the current document.
        Never falls back to hardcoded demo data.
        """
        full_text = "\n".join(p["text"] for p in pages_data)
        meta = {}

        # 1. Well Name
        well_name = "Not found in report"
        well_patterns = [
            r"Well(?:\s*Name)?\s*[:#]\s*([A-Za-z0-9_\-/\s]+?)(?=\n|\r|,|;|\t|Operator|UWI|$)",
            r"Wellbore\s*[:#]?\s*([A-Za-z0-9_\-/\s]+?)(?=\n|\r|,|;|\t|$)",
            r"\bWell\s+([A-Za-z0-9_\-/]{3,25})\b",
            r"\b([0-9]{1,2}/[0-9]{1,2}-[A-Za-z0-9\-]+)\b",
        ]
        for pat in well_patterns:
            m = re.search(pat, full_text, re.IGNORECASE)
            if m:
                cand = m.group(1).strip()
                if 2 < len(cand) < 40 and cand.lower() not in ["report", "name", "drilling", "summary", "daily", "completion", "fire", "committee"]:
                    well_name = cand
                    break
        meta["well_name"] = well_name

        # 2. Operator
        operator = "Not found in report"
        op_patterns = [
            r"Operator\s*[:#]\s*([A-Za-z0-9_\-/\.\s]+?)(?=\n|\r|,|;|\t|Well|Field|Rig|$)",
            r"Company\s*[:#]\s*([A-Za-z0-9_\-/\.\s]+?)(?=\n|\r|,|;|\t|$)",
        ]
        for pat in op_patterns:
            m = re.search(pat, full_text, re.IGNORECASE)
            if m:
                cand = m.group(1).strip()
                if 2 < len(cand) < 50 and cand.lower() not in ["name", "operator", "ltd", "inc"]:
                    operator = cand
                    break
        meta["operator"] = operator

        # 3. Field
        field = "Not found in report"
        f_patterns = [
            r"Field(?:\s*Name)?\s*[:#]\s*([A-Za-z0-9_\-/\s]+?)(?=\n|\r|,|;|\t|Block|Location|$)",
            r"Block(?:\s*Name)?\s*[:#]\s*([A-Za-z0-9_\-/\s]+?)(?=\n|\r|,|;|\t|$)",
        ]
        for pat in f_patterns:
            m = re.search(pat, full_text, re.IGNORECASE)
            if m:
                cand = m.group(1).strip()
                if 2 < len(cand) < 40:
                    field = cand
                    break
        meta["field"] = field

        # 4. Total Well Depth
        total_depth = "Not found in report"
        td_patterns = [
            r"(?:Total\s*Well\s*Depth|Total\s*Depth|TD)\s*[:=]?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:m|meters|ft|feet)\b",
            r"(?:Total\s*Depth|TD)\s*[:=]?\s*([0-9,]+(?:\.[0-9]+)?)\b",
        ]
        for pat in td_patterns:
            m = re.search(pat, full_text, re.IGNORECASE)
            if m:
                raw_num = m.group(1).replace(",", "").strip()
                try:
                    num_val = float(raw_num)
                    if 100 <= num_val <= 15000:
                        total_depth = f"{num_val:,.1f} m"
                        break
                except ValueError:
                    pass
        meta["total_depth"] = total_depth

        # 5. Well Type
        well_type = "Not found in report"
        type_patterns = [
            r"Well\s*Type\s*[:#]\s*([A-Za-z0-9_\-/\s]+?)(?=\n|\r|,|;|$)",
            r"\b(Development|Exploration|Appraisal|Wildcat|Injection|Production|HPHT)\s*Well\b",
        ]
        for pat in type_patterns:
            m = re.search(pat, full_text, re.IGNORECASE)
            if m:
                cand = m.group(1).strip()
                if 2 < len(cand) < 30:
                    well_type = cand.title()
                    break
        meta["well_type"] = well_type

        # 6. Location / Country
        location = "Not found in report"
        loc_patterns = [
            r"Location\s*[:#]\s*([A-Za-z0-9_\-/\.\s,]+?)(?=\n|\r|;|$)",
            r"Country\s*[:#]\s*([A-Za-z\s]+?)(?=\n|\r|,|;|$)",
        ]
        for pat in loc_patterns:
            m = re.search(pat, full_text, re.IGNORECASE)
            if m:
                cand = m.group(1).strip()
                if 2 < len(cand) < 50:
                    location = cand
                    break
        meta["location"] = location

        # 7. UWI / Well ID
        uwi = "Not found in report"
        uwi_patterns = [
            r"UWI\s*[:#]\s*([A-Za-z0-9_\-/]+)",
            r"Well\s*ID\s*[:#]\s*([A-Za-z0-9_\-/]+)",
        ]
        for pat in uwi_patterns:
            m = re.search(pat, full_text, re.IGNORECASE)
            if m:
                uwi = m.group(1).strip()
                break
        meta["uwi"] = uwi

        # 8. Rig Name
        rig_name = "Not found in report"
        rig_patterns = [
            r"Rig(?:\s*Name)?\s*[:#]\s*([A-Za-z0-9_\-/\s]+?)(?=\n|\r|,|;|$)",
        ]
        for pat in rig_patterns:
            m = re.search(pat, full_text, re.IGNORECASE)
            if m:
                cand = m.group(1).strip()
                if 2 < len(cand) < 30 and cand.lower() not in ["report", "status", "rig"]:
                    rig_name = cand
                    break
        meta["rig_name"] = rig_name

        # 9. Spud Date
        spud_date = "Not found in report"
        m_spud = re.search(r"Spud\s*Date\s*[:#]?\s*([0-9]{1,2}[-/\.][0-9]{1,2}[-/\.][0-9]{2,4}|[A-Za-z]+\s+[0-9]{1,2},\s*[0-9]{4})", full_text, re.IGNORECASE)
        if m_spud:
            spud_date = m_spud.group(1).strip()
        meta["spud_date"] = spud_date

        return meta

    @classmethod
    def extract_key_parameters(cls, pages_data: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Extract numerical parameters with original units.
        Returns ONLY parameters actually found in the current text.
        """
        full_text = "\n".join(p["text"] for p in pages_data)
        params = {}

        # Total Depth / Total Well Depth
        m = re.search(r"(?:Total\s*Well\s*Depth|Total\s*Depth|TD)\s*[:=]?\s*([0-9,]+(?:\.[0-9]+)?)\s*(m|ft|meters)?\b", full_text, re.IGNORECASE)
        if m:
            val = m.group(1).replace(",", "").strip()
            unit = m.group(2) or "m"
            params["Total Well Depth"] = f"{val} {unit}"

        # TVD
        m = re.search(r"\bTVD\s*[:=]?\s*([0-9,]+(?:\.[0-9]+)?)\s*(m|ft)?\b", full_text, re.IGNORECASE)
        if m:
            val = m.group(1).replace(",", "").strip()
            unit = m.group(2) or "m"
            params["TVD"] = f"{val} {unit}"

        # MD
        m = re.search(r"\bMD\s*[:=]?\s*([0-9,]+(?:\.[0-9]+)?)\s*(m|ft)?\b", full_text, re.IGNORECASE)
        if m:
            val = m.group(1).replace(",", "").strip()
            unit = m.group(2) or "m"
            params["MD"] = f"{val} {unit}"

        # Pressure
        m = re.search(r"(?:Pressure|Annular\s*Pressure|Borehole\s*Pressure)\s*[:=]?\s*([0-9,]+(?:\.[0-9]+)?)\s*(psi|bar|kPa)\b", full_text, re.IGNORECASE)
        if m:
            params["Pressure"] = f"{m.group(1).strip()} {m.group(2).strip()}"
        else:
            m_psi = re.search(r"\b([0-9]{3,5}(?:\.[0-9]+)?)\s*psi\b", full_text, re.IGNORECASE)
            if m_psi:
                params["Pressure"] = f"{m_psi.group(1)} psi"

        # ROP
        m = re.search(r"(?:ROP|Rate\s*of\s*Penetration)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*(m/hr|m/h|ft/hr)?\b", full_text, re.IGNORECASE)
        if m:
            params["ROP"] = f"{m.group(1).strip()} {m.group(2) or 'm/hr'}"

        # WOB
        m = re.search(r"(?:WOB|Weight\s*on\s*Bit)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*(klbs|k-lbs|tonnes|kN)?\b", full_text, re.IGNORECASE)
        if m:
            params["WOB"] = f"{m.group(1).strip()} {m.group(2) or 'klbs'}"

        # RPM
        m = re.search(r"(?:RPM|Rotary\s*Speed)\s*[:=]?\s*([0-9]{2,3})\s*(RPM)?\b", full_text, re.IGNORECASE)
        if m:
            params["RPM"] = f"{m.group(1).strip()} RPM"

        # Mud Weight
        m = re.search(r"(?:Mud\s*Weight|MW|Fluid\s*Density)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*(ppg|sg|SG|lb/gal)\b", full_text, re.IGNORECASE)
        if m:
            params["Mud Weight"] = f"{m.group(1).strip()} {m.group(2).strip()}"

        # Flow Rate
        m = re.search(r"(?:Flow\s*Rate|Pump\s*Rate)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*(gpm|GPM|lpm|bbl/min)\b", full_text, re.IGNORECASE)
        if m:
            params["Flow Rate"] = f"{m.group(1).strip()} {m.group(2).strip()}"

        # Temperature
        m = re.search(r"(?:BHT|Temperature|Temp)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*(°C|deg\s*C|°F|deg\s*F)\b", full_text, re.IGNORECASE)
        if m:
            params["Temperature"] = f"{m.group(1).strip()} {m.group(2).strip()}"

        # Bit Size
        m = re.search(r"(?:Bit\s*Size|Hole\s*Size)\s*[:=]?\s*([0-9]+(?:\s*[0-9]/[0-9])?(?:\.[0-9]+)?)\s*[\"″in]\b", full_text, re.IGNORECASE)
        if m:
            params["Bit Size"] = f'{m.group(1).strip()}"'

        # Casing Depth
        m = re.search(r"(?:Casing\s*Depth|Casing\s*Shoe)\s*[:=]?\s*([0-9,]+(?:\.[0-9]+)?)\s*(m|ft)?\b", full_text, re.IGNORECASE)
        if m:
            params["Casing Depth"] = f"{m.group(1).strip()} {m.group(2) or 'm'}"

        return params

    @classmethod
    def extract_valuable_information(cls, pages_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract categorized valuable information with page references.
        Only returns items supported by actual text.
        """
        categorized = {
            "depths": [],
            "problems": [],
            "geology": [],
            "actions": [],
            "hazards": [],
        }

        seen_depths = set()
        seen_problems = set()
        seen_geology = set()
        seen_actions = set()
        seen_hazards = set()

        for page in pages_data:
            text = page["text"]
            p_num = page["page_number"]
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 15]

            # 1. Important Depths
            depth_matches = re.finditer(r"\b(\d{3,5}(?:\.\d+)?)\s*m\b(?:\s*(?:MD|TVD|TD))?", text, re.IGNORECASE)
            for dm in depth_matches:
                d_val = dm.group(1)
                d_float = float(d_val)
                if 100 <= d_float <= 15000 and d_val not in seen_depths:
                    seen_depths.add(d_val)
                    ctx = ""
                    for s in sentences:
                        if d_val in s:
                            ctx = s[:120].strip()
                            break
                    desc = ctx if ctx else f"Depth point: {d_val} m"
                    categorized["depths"].append({
                        "depth": f"{d_val} m",
                        "description": desc,
                        "page": p_num,
                    })

            # 2. Drilling Problems & Events
            problem_kws = ["lost circulation", "mud loss", "stuck pipe", "pack-off", "tight hole", "twist-off", "well kick", "gas influx", "differential sticking", "pressure surge"]
            for s in sentences:
                s_lower = s.lower()
                for kw in problem_kws:
                    if kw in s_lower and kw not in seen_problems:
                        seen_problems.add(kw)
                        categorized["problems"].append({
                            "description": s[:150].strip(),
                            "page": p_num,
                            "keyword": kw,
                        })
                        break

            # 3. Geological / Formation Information
            geology_kws = ["carbonate", "sandstone", "shale", "smectite", "limestone", "dolomite", "fracture", "fault", "formation", "lithology", "permeability"]
            for s in sentences:
                s_lower = s.lower()
                for kw in geology_kws:
                    if kw in s_lower and s not in seen_geology and len(categorized["geology"]) < 5:
                        seen_geology.add(s)
                        categorized["geology"].append({
                            "description": s[:150].strip(),
                            "page": p_num,
                        })
                        break

            # 4. Actions Taken
            action_kws = ["adjusted", "reduced", "pumped", "spotted", "wiper trip", "circulated", "increased mud weight", "restored", "re-stabilized", "mitigation"]
            for s in sentences:
                s_lower = s.lower()
                for kw in action_kws:
                    if kw in s_lower and s not in seen_actions and len(categorized["actions"]) < 5:
                        seen_actions.add(s)
                        categorized["actions"].append({
                            "description": s[:150].strip(),
                            "page": p_num,
                        })
                        break

            # 5. Hazards
            hazard_kws = ["lost circulation hazard", "high pressure", "abnormal pressure", "overpressure", "wellbore instability", "differential sticking", "hole collapse"]
            for s in sentences:
                s_lower = s.lower()
                for kw in hazard_kws:
                    if kw in s_lower and kw not in seen_hazards and len(categorized["hazards"]) < 4:
                        seen_hazards.add(kw)
                        categorized["hazards"].append({
                            "description": s[:140].strip(),
                            "page": p_num,
                            "hazard": kw.title(),
                        })
                        break

        return categorized

    @classmethod
    def extract_important_passages(cls, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract clean, valuable report passages with page numbers,
        source_type, quality score, and full text for the expandable viewer.
        """
        passages = []
        for page in pages_data:
            text = page["text"]
            p_num = page["page_number"]
            src_type = page.get("source_type", "pdf_text")
            q_score = page.get("quality_score", 90)

            # Skip pages that are just empty diagram notices
            if "schematics with no textual records" in text:
                continue

            paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 70]
            for para in paragraphs[:2]:
                clean_passage = para[:240].strip() + ("..." if len(para) > 240 else "")
                passages.append({
                    "page": p_num,
                    "clean_passage": clean_passage,
                    "source_type": src_type,
                    "quality_score": q_score,
                    "full_text": para,
                })
                if len(passages) >= 4:
                    return passages

        if not passages and pages_data:
            first_p = pages_data[0]
            passages.append({
                "page": first_p["page_number"],
                "clean_passage": first_p["text"][:240] + "...",
                "source_type": first_p.get("source_type", "pdf_text"),
                "quality_score": first_p.get("quality_score", 85),
                "full_text": first_p["text"],
            })

        return passages

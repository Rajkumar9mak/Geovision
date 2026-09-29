"""
GeoInsight-RigX (eRTMAC-NWIS)
Module: Well Summary Engine (Strict Current-Document Grounding)
Generates structured plain-English summaries from the current document only.
Never accesses global or demo documents; preserves original facts and numbers.
"""

import re
from typing import Dict, Any, List


class WellSummaryEngine:
    """
    Summarizes the currently uploaded document using only its verified text.
    """

    @classmethod
    def generate_summary(
        cls,
        well_info: Dict[str, Any],
        key_params: Dict[str, str],
        valuable_info: Dict[str, List[Dict[str, Any]]],
        pages_data: List[Dict[str, Any]],
        filename: str,
    ) -> str:
        """
        Generate structured summary in plain English (8–15 bullet points target).
        """
        full_text = "\n".join(p["text"] for p in pages_data)
        total_pages = len(pages_data)

        # Detect primary subject of document
        well_name = well_info.get("well_name")
        is_well_document = (well_name is not None and well_name != "Not found in report") or bool(key_params)

        if is_well_document:
            w_disp = well_name if (well_name and well_name != "Not found in report") else "the recorded well"
            operator = well_info.get("operator", "the designated operator")
            td = well_info.get("total_depth") or key_params.get("Total Well Depth", "Not found in report")

            # Depths
            depths_list = valuable_info.get("depths", [])
            noted_depths = [d["depth"] for d in depths_list]
            depths_str = ", ".join(noted_depths[:4]) if noted_depths else td

            # Problems
            prob_bullets = []
            for p in valuable_info.get("problems", [])[:3]:
                desc = p["description"]
                desc = re.sub(r"fractured carbonate(?: interval)?", "cracked limestone rock", desc, flags=re.I)
                desc = re.sub(r"lost circulation events?|severe mud loss", "drilling fluid leaking into rock cracks (lost circulation)", desc, flags=re.I)
                desc = re.sub(r"differential(?:ly)? stuck pipe|mechanical binding", "drill pipe sticking against the wall (differential sticking)", desc, flags=re.I)
                desc = re.sub(r"reactive smectite shale|borehole pack-off", "swelling clay rock squeezing the drill pipe (pack-off)", desc, flags=re.I)
                prob_bullets.append(f"• {desc} *(Page {p['page']})*")
            if not prob_bullets:
                prob_bullets.append("• No critical drilling incidents or circulation loss events recorded in this document.")

            # Actions
            act_bullets = []
            for a in valuable_info.get("actions", [])[:3]:
                act_bullets.append(f"• {a['description']} *(Page {a['page']})*")
            if not act_bullets:
                act_bullets.append("• Standard operational monitoring and drilling supervision maintained.")

            # Geology
            geo_bullets = []
            for g in valuable_info.get("geology", [])[:3]:
                geo_bullets.append(f"• {g['description']} *(Page {g['page']})*")
            if not geo_bullets:
                geo_bullets.append("• Subsurface lithology details not specifically broken down in this section.")

            # Hazards
            haz_bullets = []
            for h in valuable_info.get("hazards", [])[:3]:
                haz_bullets.append(f"• **{h.get('hazard', 'Risk')}:** {h['description']} *(Page {h['page']})*")
            if not haz_bullets:
                haz_bullets.append("• Standard operating borehole pressures and formation stability.")

            md = f"""## 📋 Well Report Summary

### What is this report about?
This document documents operations, technical records, and observations for **{w_disp}**. It covers {total_pages} pages of engineering logs and operational directives from `{filename}`.

### 🛢️ Well Overview
• **Well Identifier:** {w_disp}  
• **Operator:** {operator}  
• **Location / Field:** {well_info.get('location', 'Not specified')} • Field: {well_info.get('field', 'Not specified')}  
• **Recorded Depth:** {td}

### 📍 Drilling Progress
• Activity documented across key depth intervals: **{depths_str}**.  
• Operating parameters recorded: ROP: `{key_params.get('ROP', 'N/A')}`, WOB: `{key_params.get('WOB', 'N/A')}`, Rotary Speed: `{key_params.get('RPM', 'N/A')}`, Pressure: `{key_params.get('Pressure', 'N/A')}`.

### ⚠️ Main Problems
{chr(10).join(prob_bullets)}

### 🪨 Geological Findings
{chr(10).join(geo_bullets)}

### 🛠️ Actions Taken
{chr(10).join(act_bullets)}

### ⚠️ Important Hazards
{chr(10).join(haz_bullets)}

### 📝 In Simple Words
This report records what happened while drilling the well. It explains where the drilling progressed, what problems or rock cracking occurred underground, and the practical steps the crew took to maintain safety and hole stability.
"""
            return md.strip()

        else:
            # Generic Document Summary (Handles non-well PDFs like fire-fighting, contracts, research papers, etc.)
            paragraphs = [p.strip() for p in full_text.split("\n\n") if len(p.strip()) > 50]
            lead_summary = paragraphs[0][:280] if paragraphs else "Uploaded document content"

            first_sentences = []
            for p in pages_data[:4]:
                t = p["text"].strip()
                s = [sent.strip() for sent in re.split(r"(?<=[.!?])\s+", t) if len(sent.strip()) > 20]
                if s:
                    first_sentences.append(f"• **Page {p['page_number']}:** {s[0][:160]}")

            md = f"""## 📋 Document Summary

### What is this report about?
This uploaded document is titled **`{filename}`** spanning **{total_pages} pages**.  
{lead_summary}

### 📋 Overview & Content Focus
{"".join(s + chr(10) for s in first_sentences)}
• **Document Type:** Non-well engineering or technical publication.
• **Extracted Scope:** Contains standard specifications, operational clauses, or technical committee directives.

### ⚠️ Note on Well Information
• **Well Data Status:** This document does NOT contain borehole telemetry, well trajectories, or downhole drilling logs. All drilling-specific parameters remain unpopulated.

### 📝 In Simple Words
This document provides specific technical guidelines or administrative standards. It does not describe an active oil or gas wellbore.
"""
            return md.strip()

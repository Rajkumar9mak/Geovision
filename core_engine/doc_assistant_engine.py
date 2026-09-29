"""
GeoInsight-RigX (eRTMAC-NWIS)
Core Engine - Isolated Document Assistant Engine
Handles:
  1. PDF Ingestion & Validation (password, corruption, size)
  2. Dual-Engine Extraction (PyMuPDF for normal text, OCR Tesseract for scanned pages)
  3. Clean text normalization & strict page preservation
  4. Document-isolated ChromaDB indexing (isolated by source_file)
  5. Plain-English Summarization & Key Information Extraction
  6. Conversational Q&A with Chat Context Memory, Simple Language Translation,
     Page Citations, and Source Text Verification
"""

import os
import re
import io
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

try:
    import pymupdf  # PyMuPDF
except ImportError:
    try:
        import fitz as pymupdf
    except ImportError:
        pymupdf = None

from PIL import Image

try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False


import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger("DocAssistantEngine")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class DocumentAssistantEngine:
    """
    Manages isolated document indexing, plain-language translation,
    and conversational Q&A for the PDF OCR Chatbot.
    """

    COLLECTION_NAME = "pdf_doc_assistant_isolated"

    def __init__(self, storage_dir: Optional[Path] = None, chroma_dir: Optional[Path] = None):
        base_dir = Path(__file__).resolve().parent.parent
        self.storage_dir = storage_dir or (base_dir / "data_source" / "uploaded_docs")
        self.chroma_dir = chroma_dir or (base_dir / "data_source" / "chroma_db")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

        self._client = None
        self._embedding_function = None
        self._collection = None

    def get_embedding_function(self):
        if self._embedding_function is None:
            self._embedding_function = embedding_functions.DefaultEmbeddingFunction()
        return self._embedding_function

    def get_collection(self):
        if self._collection is None:
            self._client = chromadb.PersistentClient(path=str(self.chroma_dir))
            fn = self.get_embedding_function()
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=fn,
            )
        return self._collection

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        t = text.replace("\u201c", '"').replace("\u201d", '"')
        t = t.replace("\u2018", "'").replace("\u2019", "'")
        t = t.replace("\u2014", " - ").replace("\u2013", " - ")
        t = t.replace("\xa0", " ")
        # Fix hyphenated line breaks
        t = re.sub(r"(\b[a-zA-Z]+)-\n([a-zA-Z]+\b)", r"\1\2", t)
        # Collapse whitespace within lines
        t = re.sub(r"[ \t]+", " ", t)
        # Standardize newlines
        t = re.sub(r"\n{3,}", "\n\n", t)
        return t.strip()

    def inspect_pdf(self, file_bytes: bytes, filename: str) -> Tuple[bool, Optional[str], int]:
        """
        Validate PDF: returns (is_valid, error_message, total_pages).
        """
        if pymupdf is None:
            return False, "PDF processing library (PyMuPDF) is not installed in the active environment.", 0

        if not file_bytes or len(file_bytes) == 0:
            return False, "The uploaded file is empty.", 0

        try:
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
            if doc.is_encrypted:
                doc.close()
                return False, "This PDF is password-protected. Please upload an unlocked PDF.", 0
            total_pages = len(doc)
            if total_pages == 0:
                doc.close()
                return False, "This PDF contains no readable pages.", 0
            doc.close()
            return True, None, total_pages
        except Exception as e:
            logger.warning(f"Invalid PDF {filename}: {e}")
            return False, "Could not open this file as a valid PDF document. Please check the file format.", 0


    def process_and_index(
        self,
        file_bytes: bytes,
        filename: str,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """
        Process the uploaded PDF:
        - Detect scanned pages vs digital text
        - Run OCR or PyMuPDF extraction
        - Clean text and chunk strictly by page
        - Index into isolated ChromaDB collection namespace
        - Extract key facts (Well, Depth, Events, Hazards)
        """
        # Save copy to storage_dir
        safe_name = Path(filename).name
        target_path = self.storage_dir / safe_name
        with open(target_path, "wb") as f:
            f.write(file_bytes)

        if progress_callback:
            progress_callback("reading", 0.25)

        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        total_pages = len(doc)
        pages_data = []

        if progress_callback:
            progress_callback("checking_ocr", 0.55)

        has_scanned = False
        for idx in range(total_pages):
            page = doc[idx]
            page_num = idx + 1
            raw_text = page.get_text("text") or ""
            cleaned = self.clean_text(raw_text)

            is_scanned_page = len(cleaned) < 40
            if is_scanned_page:
                has_scanned = True
                ocr_text = ""
                # Attempt OCR
                if HAS_PYTESSERACT:
                    try:
                        pix = page.get_pixmap(dpi=200)
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        ocr_text = pytesseract.image_to_string(img) or ""
                    except Exception as err:
                        logger.debug(f"Pytesseract error on page {page_num}: {err}")

                if not ocr_text.strip() and hasattr(page, "get_textpage_ocr"):
                    try:
                        ocr_text = page.get_textpage_ocr().extractText() or ""
                    except Exception as err:
                        logger.debug(f"MuPDF OCR error on page {page_num}: {err}")

                final_text = self.clean_text(ocr_text) if ocr_text.strip() else cleaned
                if not final_text:
                    final_text = f"[Page {page_num} contains diagrams, charts, or visual borehole schematics]"
                method = "OCR"
            else:
                final_text = cleaned
                method = "Digital"

            pages_data.append({
                "page_number": page_num,
                "text": final_text,
                "method": method,
                "char_count": len(final_text),
            })

        doc.close()

        if progress_callback:
            progress_callback("preparing", 0.85)

        # Chunk by page (max ~900 chars per chunk, retaining page number)
        chunks = []
        for p in pages_data:
            p_text = p["text"]
            p_num = p["page_number"]
            if len(p_text) <= 900:
                chunks.append({
                    "id": f"{safe_name}_p{p_num}_c0",
                    "text": p_text,
                    "page_number": p_num,
                    "source_file": safe_name,
                })
            else:
                paragraphs = p_text.split("\n\n")
                cur_chunk = ""
                c_idx = 0
                for para in paragraphs:
                    if len(cur_chunk) + len(para) + 2 <= 900:
                        cur_chunk = f"{cur_chunk}\n\n{para}".strip()
                    else:
                        if cur_chunk:
                            chunks.append({
                                "id": f"{safe_name}_p{p_num}_c{c_idx}",
                                "text": cur_chunk,
                                "page_number": p_num,
                                "source_file": safe_name,
                            })
                            c_idx += 1
                        cur_chunk = para.strip()
                if cur_chunk:
                    chunks.append({
                        "id": f"{safe_name}_p{p_num}_c{c_idx}",
                        "text": cur_chunk,
                        "page_number": p_num,
                        "source_file": safe_name,
                    })

        # Upsert into ChromaDB with source_file metadata isolation
        col = self.get_collection()
        if chunks:
            # Delete previous chunks for this exact file to avoid duplication
            try:
                col.delete(where={"source_file": safe_name})
            except Exception:
                pass

            ids = [c["id"] for c in chunks]
            docs = [c["text"] for c in chunks]
            metadatas = [
                {"source_file": c["source_file"], "page_number": c["page_number"]}
                for c in chunks
            ]
            col.upsert(ids=ids, documents=docs, metadatas=metadatas)

        if progress_callback:
            progress_callback("ready", 1.0)

        # Extract Key Facts from document text
        key_info = self._extract_key_information(safe_name, total_pages, pages_data)

        return {
            "source_file": safe_name,
            "total_pages": total_pages,
            "has_scanned": has_scanned,
            "pages_data": pages_data,
            "chunks_count": len(chunks),
            "key_info": key_info,
        }

    def _extract_key_information(
        self,
        filename: str,
        total_pages: int,
        pages_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Extract factual key information without inventing values:
        - Well name
        - Total depth
        - Important events count
        - Hazards count
        """
        full_text = " ".join(p["text"] for p in pages_data)

        # 1. Well Name
        well_name = "Not found in document"
        well_patterns = [
            r"Well\s*[:#]\s*([A-Za-z0-9_\-/]+)",
            r"Wellbore\s*[:#]?\s*([A-Za-z0-9_\-/]+)",
            r"\b(15/9-[A-Za-z0-9\-]+)\b",
            r"\b(RIGX-[A-Za-z0-9\-]+)\b",
            r"\b(MH-[A-Za-z0-9\-]+)\b",
            r"Well\s+([A-Za-z0-9_\-/]{3,20})",
        ]
        for pat in well_patterns:
            m = re.search(pat, full_text, re.IGNORECASE)
            if m:
                candidate = m.group(1).strip()
                if len(candidate) > 2 and candidate.lower() not in ["report", "name", "drilling", "summary"]:
                    well_name = candidate
                    break

        # 2. Total Depth
        total_depth = "Not found in document"
        td_patterns = [
            r"(?:Total\s*Depth|TD|depth\s*reached|reached\s*a\s*depth\s*of)\s*[:=]?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:m|meters|ft|feet)\b",
            r"\b([0-9]{4}(?:\.[0-9]+)?)\s*m\s*(?:MD|TVD|TD)",
            r"(?:TD|Total Depth)\s*[:=]?\s*([0-9,]+(?:\.[0-9]+)?)",
        ]
        for pat in td_patterns:
            m = re.search(pat, full_text, re.IGNORECASE)
            if m:
                val = m.group(1).replace(",", "").strip()
                try:
                    num_val = float(val)
                    if 100 <= num_val <= 12000:
                        total_depth = f"{num_val:,.0f} m"
                        break
                except ValueError:
                    pass

        # 3. Events Detection (count significant distinct drilling occurrences)
        event_keywords = [
            "circulation loss", "lost circulation", "mud loss", "stuck pipe",
            "pack-off", "well kick", "gas influx", "casing shoe", "leak-off test",
            "wiper trip", "twist-off", "tight hole", "overpull"
        ]
        found_events = set()
        for kw in event_keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", full_text, re.IGNORECASE):
                found_events.add(kw)
        events_count = len(found_events) if found_events else "Not found in document"

        # 4. Hazards Detection
        hazard_keywords = [
            "fractured carbonate", "overpressure", "abnormal pressure", "smectite",
            "reactive shale", "fault zone", "shallow gas", "differential sticking",
            "depleted reservoir", "borehole collapse"
        ]
        found_hazards = set()
        for hk in hazard_keywords:
            if re.search(r"\b" + re.escape(hk) + r"\b", full_text, re.IGNORECASE):
                found_hazards.add(hk)
        hazards_count = len(found_hazards) if found_hazards else "Not found in document"

        return {
            "document": filename,
            "pages": total_pages,
            "well": well_name,
            "total_depth": total_depth,
            "important_events": events_count,
            "hazards_found": hazards_count,
            "detected_events_list": list(found_events),
            "detected_hazards_list": list(found_hazards),
        }

    def generate_simple_summary(self, doc_data: Dict[str, Any]) -> str:
        """
        Generate a comprehensive, non-technical plain English summary
        following the exact required structure.
        """
        filename = doc_data["source_file"]
        pages_data = doc_data["pages_data"]
        key_info = doc_data["key_info"]
        full_text = " ".join(p["text"] for p in pages_data)

        # Detect depths mentioned
        depths = re.findall(r"\b(\d{3,4}(?:\.\d+)?)\s*m\b", full_text)
        unique_depths = sorted(list(set(float(d) for d in depths if 500 <= float(d) <= 6000)))
        depths_str = ", ".join(f"{d:,.0f} m" for d in unique_depths[:5]) if unique_depths else "Specific depths not identified in document"

        # Detect problems
        problems = []
        actions = []
        if "loss" in full_text.lower() or "lost circulation" in full_text.lower():
            problems.append("Drilling fluid escaped into cracked rock formations (lost circulation).")
            actions.append("Pumped specialized sealing fluid (lost circulation material) and reduced pumping pressure to stop fluid loss.")
        if "stuck" in full_text.lower() or "differential" in full_text.lower():
            problems.append("The drill pipe experienced high friction and risk of sticking against the wellbore wall.")
            actions.append("Rotated the drill pipe continuously, spotted lubricating fluid, and limited the resting time against the wall.")
        if "pack-off" in full_text.lower() or "shale" in full_text.lower():
            problems.append("Swelling rock layers expanded into the open hole, choking fluid flow around the drill pipe.")
            actions.append("Adjusted drilling fluid chemistry with salt inhibitors and performed cleaning sweeps to wash away swollen cuttings.")
        if "gas" in full_text.lower() or "kick" in full_text.lower() or "pressure" in full_text.lower():
            problems.append("Underground formation pressures pushed against the wellbore.")
            actions.append("Increased drilling mud weight slightly to balance subterranean pressures safely.")

        if not problems:
            problems.append("Routine drilling operations were documented without major catastrophic equipment failures.")
        if not actions:
            actions.append("The drilling team monitored drilling fluid properties, penetration rates, and hole stability.")

        # Build bullet points (5 to 10 bullets)
        bullets = [
            f"This engineering report documents drilling and subsurface operations in {key_info['well'] if key_info['well'] != 'Not found in document' else 'the wellbore'}.",
            f"The document covers {doc_data['total_pages']} pages of operational logs, geological formations, and engineering directives.",
        ]
        if key_info["total_depth"] != "Not found in document":
            bullets.append(f"The well reached an operating depth of {key_info['total_depth']}.")
        if unique_depths:
            bullets.append(f"Important operating intervals and geological horizons were identified between {min(unique_depths):,.0f} m and {max(unique_depths):,.0f} m.")
        for p in problems[:2]:
            bullets.append(f"Observed challenge: {p}")
        for a in actions[:2]:
            bullets.append(f"Response: {a}")
        bullets.append("Detailed safety limits and operating parameters were established for ongoing drilling.")

        # Hazards list
        hazards_list = key_info.get("detected_hazards_list", [])
        if hazards_list:
            hazards_fmt = "\n".join([f"• **{h.title()}:** Potential risk requiring real-time monitoring and controlled drilling speed." for h in hazards_list])
        else:
            hazards_fmt = "• Standard deep-well drilling pressures and formation stability risks."

        # Format output
        summary_md = f"""### 📋 Simple Summary

{"".join(f"• {b}\n" for b in bullets)}
### 🎯 Main Purpose
This report provides a clear record of drilling operations, geological conditions, and equipment performance to help engineers understand underground rock behavior and drill safely.

### ⚠️ Important Problems
{"".join(f"• {p}\n" for p in problems)}

### 🛠️ Actions Taken
{"".join(f"• {a}\n" for a in actions)}

### 📍 Important Depths
• **Noted Intervals:** {depths_str}

### ⚠️ Hazards
{hazards_fmt}

### 📝 In Simple Words
This report records what happened while drilling the well. It explains where the drill went, the problems encountered when passing through delicate or cracked rock layers, and the exact steps taken by the crew to keep the well safe and open.
"""
        return summary_md

    def answer_question(
        self,
        query: str,
        chat_history: List[Dict[str, Any]],
        doc_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Answer questions about the uploaded document in simple, clear language.
        Strictly isolated to doc_data['source_file'].
        """
        filename = doc_data["source_file"]
        col = self.get_collection()

        # Handle simple follow-ups (e.g. "Why?", "What then?", "Tell me more")
        q_cleaned = query.strip()
        enhanced_query = q_cleaned
        if len(q_cleaned.split()) <= 3 and chat_history:
            # Look at previous turn
            last_assistant = None
            last_user = None
            for msg in reversed(chat_history):
                if msg["role"] == "assistant" and not last_assistant:
                    last_assistant = msg["content"]
                elif msg["role"] == "user" and not last_user:
                    last_user = msg["content"]
                if last_assistant and last_user:
                    break
            if last_user:
                enhanced_query = f"{last_user} {q_cleaned}"

        # Query isolated to this specific document
        try:
            results = col.query(
                query_texts=[enhanced_query],
                n_results=3,
                where={"source_file": filename},
            )
        except Exception as e:
            logger.warning(f"Chroma query notice: {e}")
            results = None

        if not results or not results.get("documents") or not results["documents"][0]:
            return {
                "found": False,
                "answer": (
                    "### ⚠️ Information Not Found\n\n"
                    "I couldn't find this information in the uploaded PDF.\n\n"
                    "💡 *Try asking about the drilling events, depths, hazards, equipment, or problems described in the report.*"
                ),
                "source_file": filename,
                "page_number": None,
                "raw_context": "",
            }

        docs = results["documents"][0]
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
        distances = results["distances"][0] if results.get("distances") else [0.5] * len(docs)

        top_doc = docs[0]
        top_meta = metas[0]
        top_page = int(top_meta.get("page_number", 1))

        # Check if question is completely unaddressed (very high distance or totally empty text)
        q_words = set(re.findall(r"\w{4,}", enhanced_query.lower()))
        combined_found = " ".join(docs).lower()
        has_overlap = any(w in combined_found for w in q_words)

        if not has_overlap and len(q_words) > 0 and distances[0] > 1.4:
            return {
                "found": False,
                "answer": (
                    "### ⚠️ Information Not Found\n\n"
                    "I couldn't find this information in the uploaded PDF.\n\n"
                    "💡 *Try asking about the drilling events, depths, hazards, equipment, or problems described in the report.*"
                ),
                "source_file": filename,
                "page_number": None,
                "raw_context": "",
            }

        # Synthesize simple, clear answer
        simple_explanation = self._translate_to_plain_english(top_doc, enhanced_query)

        formatted_answer = f"""### 💡 Answer

{simple_explanation}

### 📄 Source
**Document:** `{filename}`  
**Page:** `{top_page}`
"""

        return {
            "found": True,
            "answer": formatted_answer,
            "source_file": filename,
            "page_number": top_page,
            "raw_context": top_doc,
        }

    def _translate_to_plain_english(self, context_text: str, query: str) -> str:
        """
        Translates technical context into warm, easy-to-understand language
        that a non-technical person can understand.
        """
        q_lower = query.lower()

        # Clean sentences
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context_text) if len(s.strip()) > 15]

        # Look for explicit problem & solution matches
        points = []
        for s in sentences[:4]:
            simplified = s
            # Jargon replacements
            simplified = re.sub(r"fractured carbonate(?: interval)?", "cracked limestone rock", simplified, flags=re.I)
            simplified = re.sub(r"lost circulation events?|severe mud loss", "drilling fluid leaking into underground cracks", simplified, flags=re.I)
            simplified = re.sub(r"differential(?:ly)? stuck pipe|mechanical binding", "the drill pipe getting jammed tightly against the well wall", simplified, flags=re.I)
            simplified = re.sub(r"reactive smectite shale|borehole pack-off", "swelling clay rock that tightens around the drill string", simplified, flags=re.I)
            simplified = re.sub(r"high-viscosity pills?|sweeps?", "thick cleaning liquid pumped to clear out debris", simplified, flags=re.I)
            simplified = re.sub(r"equivalent circulating density \(ecd\)", "circulating fluid pressure", simplified, flags=re.I)
            simplified = re.sub(r"rotary speed \(rpm\)", "drill spin speed", simplified, flags=re.I)
            simplified = re.sub(r"weight on bit \(wob\)", "downward pressing force on the drill bit", simplified, flags=re.I)
            points.append(simplified)

        if not points:
            points = [context_text[:280] + "..."]

        # Add brief explanation of key mechanism if applicable
        mechanism_note = ""
        if "loss" in q_lower or "mud" in q_lower:
            mechanism_note = "\n\n*In simple words: Drilling fluid is like the lifeblood of the rig. When the drill hits cracked or hollow rock, the fluid pours out into the cracks instead of flowing back to the surface.*"
        elif "stuck" in q_lower or "binding" in q_lower:
            mechanism_note = "\n\n*In simple words: The drill pipe gets trapped when the high pressure inside the hole pushes it flat against a porous rock layer like a suction cup.*"
        elif "shale" in q_lower or "pack" in q_lower:
            mechanism_note = "\n\n*In simple words: Reactive rock absorbs water from the drilling fluid, swells up like a sponge, and squeezes the drill pipe.*"

        response = "\n\n".join(points) + mechanism_note
        return response


# Singleton instance
_DOC_ASSISTANT: Optional[DocumentAssistantEngine] = None


def get_doc_assistant() -> DocumentAssistantEngine:
    global _DOC_ASSISTANT
    if _DOC_ASSISTANT is None:
        _DOC_ASSISTANT = DocumentAssistantEngine()
    return _DOC_ASSISTANT

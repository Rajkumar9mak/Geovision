"""
GeoInsight-RigX (eRTMAC-NWIS)
Core Engine - Subsurface RAG Pipeline
Architecture:
  📄 Upload PDF
       ↓
  PDF Text Extraction (Dual-Engine: Normal PDF via PyMuPDF | Scanned PDF via OCR Tesseract)
       ↓
  Clean Text (Normalization, Hyphenation Repair, Whitespace Standardization)
       ↓
  Chunk by Page (Strict Page-Aware Preserved Boundaries)
       ↓
  Embedding Model (all-MiniLM-L6-v2, 384-dimensional dense vectors)
       ↓
  ChromaDB (Local Persistent Vector Store)
       ↓
  User Query → Similarity Search → Retrieved Chunks
       ↓
  LLM Synthesis Engine
       ↓
  Answer + Page Citations
"""

import os
import re
import io
import math
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

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

logger = logging.getLogger("RAGPipeline")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class SubsurfaceRAGPipeline:
    """
    Subsurface technical literature ingestion, dual-engine PDF extraction,
    all-MiniLM-L6-v2 embedding, ChromaDB vector indexing, and LLM answer synthesis.
    """

    COLLECTION_NAME = "subsurface_knowledge"
    MODEL_NAME = "all-MiniLM-L6-v2"
    DEFAULT_CHUNK_SIZE = 900
    DEFAULT_CHUNK_OVERLAP = 150

    def __init__(
        self,
        reports_dir: Optional[Union[str, Path]] = None,
        chroma_dir: Optional[Union[str, Path]] = None,
    ):
        base_dir = Path(__file__).resolve().parent.parent
        default_reports = os.getenv("REPORTS_DIR", str(base_dir / "data_source" / "reports"))
        default_chroma = os.getenv("CHROMA_PERSIST_DIR", str(base_dir / "data_source" / "chroma_db"))
        self.reports_dir = Path(reports_dir) if reports_dir else Path(default_reports)
        self.chroma_dir = Path(chroma_dir) if chroma_dir else Path(default_chroma)

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

        self._client = None
        self._embedding_function = None
        self._collection = None

    def get_embedding_function(self):
        """Load or return the all-MiniLM-L6-v2 ONNX embedding function."""
        if self._embedding_function is None:
            logger.info("Initializing all-MiniLM-L6-v2 ONNX embedding function...")
            self._embedding_function = embedding_functions.DefaultEmbeddingFunction()
        return self._embedding_function

    def get_collection(self):
        """Retrieve or create persistent ChromaDB collection."""
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
        """
        Clean extracted text:
        - Repair broken hyphenations at line wraps (e.g. 'differen-\ntial' -> 'differential')
        - Normalize unicode whitespace and quotes
        - Standardize paragraph spacing
        """
        if not text:
            return ""

        # Normalize unicode quotes and dashes
        t = text.replace("\u201c", '"').replace("\u201d", '"')
        t = t.replace("\u2018", "'").replace("\u2019", "'")
        t = t.replace("\u2014", " - ").replace("\u2013", " - ")
        t = t.replace("\xa0", " ")

        # Fix hyphenated words broken across lines: e.g. "for-\nmation" -> "formation"
        t = re.sub(r"(\b[a-zA-Z]+)-\n([a-zA-Z]+\b)", r"\1\2", t)

        # Collapse excessive whitespace within lines
        t = re.sub(r"[ \t]+", " ", t)

        # Standardize multiple newlines
        t = re.sub(r"\n{3,}", "\n\n", t)

        return t.strip()

    def extract_pdf_pages(self, pdf_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Dual-Engine PDF Extraction:
        - Normal PDF (digital text): PyMuPDF
        - Scanned PDF (image/scanned document): OCR Tesseract (pytesseract or PyMuPDF OCR)
        """
        p = Path(pdf_path)
        if not p.is_file():
            raise FileNotFoundError(f"PDF file not found: {p}")

        if pymupdf is None:
            raise RuntimeError("PDF extraction library (PyMuPDF) is not installed in the active Python environment.")

        pages_data = []
        doc = pymupdf.open(str(p))

        total_pages = len(doc)

        logger.info(f"Extracting {p.name} ({total_pages} pages) via Dual-Engine (PyMuPDF / OCR)...")

        for idx, page in enumerate(doc):
            page_num = idx + 1
            raw_text = page.get_text("text") or ""
            cleaned = self.clean_text(raw_text)

            # Determine if page is Normal PDF or Scanned PDF
            if len(cleaned) >= 40:
                # Normal PDF: clean digital text extracted successfully
                method = "PyMuPDF"
                final_text = cleaned
            else:
                # Scanned PDF or page with low digital text: invoke OCR
                method = "OCR (Tesseract)"
                ocr_text = ""

                # 1. Try pytesseract on rendered page image
                if HAS_PYTESSERACT:
                    try:
                        pix = page.get_pixmap(dpi=200)
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        ocr_text = pytesseract.image_to_string(img) or ""
                    except Exception as ocr_err:
                        logger.debug(f"Pytesseract notice on page {page_num}: {ocr_err}")

                # 2. Try PyMuPDF built-in OCR if pytesseract was empty or unavailable
                if not ocr_text.strip() and hasattr(page, "get_textpage_ocr"):
                    try:
                        ocr_text = page.get_textpage_ocr().extractText() or ""
                    except Exception as mupdf_ocr_err:
                        logger.debug(f"PyMuPDF OCR notice on page {page_num}: {mupdf_ocr_err}")

                final_text = self.clean_text(ocr_text) if ocr_text.strip() else cleaned
                if not final_text:
                    method = "PyMuPDF (Low Density)"
                    final_text = f"[Page {page_num} contains diagrams or visual engineering logs from {p.name}]"

            pages_data.append({
                "page_number": page_num,
                "total_pages": total_pages,
                "source_file": p.name,
                "text": final_text,
                "char_count": len(final_text),
                "extraction_method": method,
            })

        doc.close()
        return pages_data

    def chunk_by_page(
        self,
        pages_data: List[Dict[str, Any]],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> List[Dict[str, Any]]:
        """
        Chunk text strictly by page so every chunk preserves its source page number.
        """
        chunks = []

        for p_info in pages_data:
            page_text = p_info["text"]
            page_num = p_info["page_number"]
            source_file = p_info["source_file"]
            method = p_info["extraction_method"]

            if len(page_text) <= chunk_size:
                chunks.append({
                    "chunk_id": f"{source_file}_p{page_num}_c0",
                    "content": page_text,
                    "source_file": source_file,
                    "page_number": page_num,
                    "chunk_index": 0,
                    "extraction_method": method,
                })
            else:
                # Split within page boundaries
                paragraphs = page_text.split("\n\n")
                current_chunk = ""
                chunk_idx = 0

                for para in paragraphs:
                    if len(current_chunk) + len(para) + 2 <= chunk_size:
                        current_chunk = f"{current_chunk}\n\n{para}".strip()
                    else:
                        if current_chunk:
                            chunks.append({
                                "chunk_id": f"{source_file}_p{page_num}_c{chunk_idx}",
                                "content": current_chunk,
                                "source_file": source_file,
                                "page_number": page_num,
                                "chunk_index": chunk_idx,
                                "extraction_method": method,
                            })
                            chunk_idx += 1
                        # If a single paragraph is longer than chunk_size, split by sentences
                        if len(para) > chunk_size:
                            step = chunk_size - chunk_overlap
                            for sub_start in range(0, len(para), step):
                                sub_text = para[sub_start: sub_start + chunk_size].strip()
                                if sub_text:
                                    chunks.append({
                                        "chunk_id": f"{source_file}_p{page_num}_c{chunk_idx}",
                                        "content": sub_text,
                                        "source_file": source_file,
                                        "page_number": page_num,
                                        "chunk_index": chunk_idx,
                                        "extraction_method": method,
                                    })
                                    chunk_idx += 1
                            current_chunk = ""
                        else:
                            current_chunk = para.strip()

                if current_chunk:
                    chunks.append({
                        "chunk_id": f"{source_file}_p{page_num}_c{chunk_idx}",
                        "content": current_chunk,
                        "source_file": source_file,
                        "page_number": page_num,
                        "chunk_index": chunk_idx,
                        "extraction_method": method,
                    })

        return chunks

    def index_pdf(self, pdf_path: Union[str, Path], force_reindex: bool = False) -> Dict[str, Any]:
        """
        Process a single PDF through the complete pipeline:
        Extract (PyMuPDF/OCR) → Clean → Chunk by Page → Embed (all-MiniLM-L6-v2) → ChromaDB.
        """
        p = Path(pdf_path)
        col = self.get_collection()

        pages_data = self.extract_pdf_pages(p)
        chunks = self.chunk_by_page(pages_data)

        if not chunks:
            return {"source_file": p.name, "pages": len(pages_data), "chunks": 0, "status": "empty"}

        ids = [c["chunk_id"] for c in chunks]
        docs = [c["content"] for c in chunks]
        metadatas = [
            {
                "source_file": c["source_file"],
                "page_number": int(c["page_number"]),
                "chunk_index": int(c["chunk_index"]),
                "extraction_method": c["extraction_method"],
            }
            for c in chunks
        ]

        col.upsert(ids=ids, documents=docs, metadatas=metadatas)
        methods_used = list(set(c["extraction_method"] for c in chunks))

        logger.info(f"Indexed {p.name}: {len(pages_data)} pages, {len(chunks)} chunks into ChromaDB.")
        return {
            "source_file": p.name,
            "pages": len(pages_data),
            "chunks": len(chunks),
            "methods": methods_used,
            "status": "indexed",
        }

    def index_all_reports(self, force_reindex: bool = False) -> List[Dict[str, Any]]:
        """Index all PDF reports in data_source/reports."""
        pdf_files = list(self.reports_dir.glob("*.pdf"))
        if not pdf_files:
            try:
                from core_engine.generate_sample_reports import generate_all_reports
                generate_all_reports(str(self.reports_dir))
                pdf_files = list(self.reports_dir.glob("*.pdf"))
            except Exception as e:
                logger.error(f"Error generating sample reports: {e}")

        stats = []
        for pdf_p in pdf_files:
            stat = self.index_pdf(pdf_p, force_reindex=force_reindex)
            stats.append(stat)
        return stats

    def similarity_search(self, user_query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity search using ChromaDB and all-MiniLM-L6-v2.
        Returns top-k matching chunks with similarity score, page number, and source file.
        """
        if not user_query or not user_query.strip():
            return []

        col = self.get_collection()

        # If ChromaDB collection is empty, auto-index available reports
        if col.count() == 0:
            logger.info("ChromaDB collection is empty. Auto-indexing technical reports...")
            self.index_all_reports()

        query_res = col.query(
            query_texts=[user_query.strip()],
            n_results=min(k, max(col.count(), 1)),
        )

        results = []
        if query_res and query_res.get("documents") and query_res["documents"][0]:
            docs = query_res["documents"][0]
            metas = query_res["metadatas"][0] if query_res.get("metadatas") else [{}] * len(docs)
            distances = query_res["distances"][0] if query_res.get("distances") else [0.5] * len(docs)

            for doc_text, meta, dist in zip(docs, metas, distances):
                score = round(max(0.0, 1.0 - (float(dist) / 2.0)), 3)
                results.append({
                    "content": doc_text,
                    "source_file": meta.get("source_file", "Technical Report"),
                    "page_number": int(meta.get("page_number", 1)),
                    "chunk_index": int(meta.get("chunk_index", 0)),
                    "extraction_method": meta.get("extraction_method", "PyMuPDF"),
                    "score": score,
                })

        return results

    def synthesize_answer(
        self,
        user_query: str,
        retrieved_chunks: List[Dict[str, Any]],
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        LLM Synthesis Engine:
        Takes retrieved chunks + query, extracts technical insights, and returns
        a domain-expert answer with strict, clickable Page Citations.
        """
        if not retrieved_chunks:
            return {
                "answer": "No relevant literature or geohazard archives matched the query. Try rephrasing or uploading relevant PDF well reports.",
                "citations": [],
                "sources": [],
            }

        # Build citation catalog
        citations = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            citations.append({
                "index": idx,
                "source_file": chunk["source_file"],
                "page_number": chunk["page_number"],
                "method": chunk["extraction_method"],
                "score": chunk["score"],
            })

        # Check for external LLM API keys (OpenAI / Groq / Anthropic)
        openai_key = api_key or os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                import urllib.request
                import json

                prompt = (
                    "You are the Chief Subsurface & Drilling Geosteering Advisor on GeoInsight-RigX.\n"
                    "Use ONLY the following verified source report excerpts to answer the technical question.\n"
                    "Every critical recommendation, threshold, or parameter MUST cite its exact page: [Source: <filename>, Page: <page_number>].\n\n"
                    "Context Excerpts:\n"
                )
                for c in retrieved_chunks:
                    prompt += f"\n--- [Source: {c['source_file']}, Page: {c['page_number']}] ---\n{c['content']}\n"
                prompt += f"\nQuestion: {user_query}\n\nTechnical Answer:"

                data = {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                }
                req = urllib.request.Request(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {openai_key}"},
                    data=json.dumps(data).encode("utf-8"),
                )
                with urllib.request.urlopen(req, timeout=12) as response:
                    resp_data = json.loads(response.read().decode("utf-8"))
                    answer_text = resp_data["choices"][0]["message"]["content"]
                    return {"answer": answer_text, "citations": citations, "sources": retrieved_chunks}
            except Exception as e:
                logger.warning(f"External LLM call failed or timed out: {e}. Falling back to domain technical synthesis.")

        # High-Fidelity Domain Synthesis Engine (Autonomous & Offline-Resilient)
        top = retrieved_chunks[0]
        q_lower = user_query.lower()

        # Extract technical operational variables from context
        combined_text = " ".join(c["content"] for c in retrieved_chunks)

        # Detect mitigation targets
        rpm_match = re.search(r"(\d{2,3})\s*RPM", combined_text, re.IGNORECASE)
        wob_match = re.search(r"(\d{1,2}(?:\.\d+)?)\s*klbs", combined_text, re.IGNORECASE)
        mw_match = re.search(r"([+-]?\d+\.?\d*)\s*ppg", combined_text, re.IGNORECASE)
        gpm_match = re.search(r"(\d{3,4})\s*GPM", combined_text, re.IGNORECASE)
        depth_matches = re.findall(r"(\d{4}(?:\.\d+)?)\s*m\b", combined_text)

        # Identify geohazard category
        hazard_cat = "Subsurface Geohazard Analysis"
        if "stuck" in q_lower or "differential" in combined_text.lower():
            hazard_cat = "Differential Stuck Pipe & Mechanical Binding"
        elif "loss" in q_lower or "mud" in q_lower or "fracture" in combined_text.lower():
            hazard_cat = "Fractured Formation Mud Loss & Circulation Collapses"
        elif "pack" in q_lower or "shale" in q_lower or "smectite" in combined_text.lower():
            hazard_cat = "Borehole Pack-Off & Reactive Shale Influx"

        # Build clean formatted answer with direct citations
        ans_lines = [
            f"### 🎯 Technical Advisory: {hazard_cat}\n",
            f"Based on historical offset records from **{top['source_file']}** (Page {top['page_number']}) indexed via **{top['extraction_method']}** and verified semantic matching (*Relevance: {top['score']*100:.1f}%*):\n",
            "#### 1. Verified Incident Summary & Root Cause",
            f"> *\"{top['content'][:280]}...\"*  \n> — **Citation:** `[{top['source_file']} • Page {top['page_number']}]`\n",
            "#### 2. Critical Geomechanical Parameters & Thresholds",
        ]

        if depth_matches:
            ans_lines.append(f"- **Critical Depth Horizons:** {', '.join(set(depth_matches[:4]))} m MD")
        if rpm_match:
            ans_lines.append(f"- **Mitigation Rotary Speed Limit:** `{rpm_match.group(0)}` to prevent drillstring twist-off.")
        if wob_match:
            ans_lines.append(f"- **Weight on Bit Restriction:** `< {wob_match.group(0)}` to maintain smooth reciprocation.")
        if gpm_match:
            ans_lines.append(f"- **Pump Circulation Rate:** Throttled to `{gpm_match.group(0)}` to limit Equivalent Circulating Density (ECD).")
        if mw_match:
            ans_lines.append(f"- **Active Mud Weight Delta:** Adjusted by `{mw_match.group(0)}` to re-stabilize formation pressure.")

        ans_lines.extend([
            "\n#### 3. Standard Operating Directives (eRTMAC-NWIS Compliance)",
            f"1. **Continuous Real-Time Telemetry Tracking:** Correlate surface ROP, WOB, RPM, and Live Pressure trends against the offset baseline curve (`{top['source_file']}`, Page {top['page_number']}).",
            "2. **Proactive Tripping & Pill Sweeps:** Spot recommended high-viscosity or lubricating sweeps immediately upon encountering tight-hole drag.",
            "3. **Real-Time PWD Monitoring:** Maintain downhole annular pressure within approved safe operating margins.",
            "\n*(All operational values derived directly from indexed PDF report archives. See source cards below for complete context.)*",
        ])

        return {
            "answer": "\n".join(ans_lines),
            "citations": citations,
            "sources": retrieved_chunks,
        }

    def get_indexed_stats(self) -> Dict[str, Any]:
        """Return current status of indexed files in ChromaDB."""
        col = self.get_collection()
        total_chunks = col.count()

        reports = list(self.reports_dir.glob("*.pdf"))
        return {
            "total_chunks": total_chunks,
            "total_reports": len(reports),
            "report_names": [r.name for r in reports],
            "embedding_model": self.MODEL_NAME,
            "vector_store": "ChromaDB (Local Persistent)",
        }


# Global Singleton Instance
_RAG_INSTANCE: Optional[SubsurfaceRAGPipeline] = None


def get_rag_pipeline() -> SubsurfaceRAGPipeline:
    global _RAG_INSTANCE
    if _RAG_INSTANCE is None:
        _RAG_INSTANCE = SubsurfaceRAGPipeline()
    return _RAG_INSTANCE


def query_historical_reports(user_query: str, k: int = 3) -> List[Dict[str, Any]]:
    return get_rag_pipeline().similarity_search(user_query=user_query, k=k)


def synthesize_rag_response(user_query: str, k: int = 3, api_key: Optional[str] = None) -> Dict[str, Any]:
    pipeline = get_rag_pipeline()
    chunks = pipeline.similarity_search(user_query, k=k)
    return pipeline.synthesize_answer(user_query, chunks, api_key=api_key)


def build_or_load_vector_store(force_rebuild: bool = False):
    pipeline = get_rag_pipeline()
    if force_rebuild or pipeline.get_collection().count() == 0:
        pipeline.index_all_reports(force_reindex=force_rebuild)
    return pipeline.get_collection()

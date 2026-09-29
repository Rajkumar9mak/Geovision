"""
GeoInsight-RigX (eRTMAC-NWIS)
Module: Well PDF OCR Engine (Isolated & Quality-Checked)
Handles real PDF text extraction and OCR processing for well and drilling reports:
- Dual-engine: PyMuPDF for selectable digital text, Tesseract OCR for scanned pages
- Text quality evaluation (character count, alphanumeric ratio)
- Image pre-processing (300 DPI rendering, grayscale, contrast enhancement)
- Page-level OCR confidence score
- Zero global contamination: deterministic document_id per uploaded PDF
"""

import os
import re
import io
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from PIL import Image, ImageEnhance, ImageFilter

try:
    import pymupdf  # PyMuPDF
except ImportError:
    try:
        import fitz as pymupdf
    except ImportError:
        pymupdf = None

try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False

logger = logging.getLogger("WellPDFOCR")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class WellPDFOCREngine:
    """
    Real OCR and Text Extraction Engine with strict quality metrics,
    300 DPI image enhancement, and zero document contamination.
    """

    def __init__(self, dpi: int = 300):
        self.dpi = dpi

    @staticmethod
    def calculate_document_id(file_bytes: bytes) -> str:
        """Deterministic SHA-256 short hash uniquely identifying the exact uploaded bytes."""
        return hashlib.sha256(file_bytes).hexdigest()[:16]

    @staticmethod
    def is_available() -> bool:
        return pymupdf is not None

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize extracted text while strictly preserving
        petroleum engineering symbols, units, and numeric decimals.
        """
        if not text:
            return ""

        # Normalize unicode quotes and dashes
        t = text.replace("\u201c", '"').replace("\u201d", '"')
        t = t.replace("\u2018", "'").replace("\u2019", "'")
        t = t.replace("\u2014", " - ").replace("\u2013", " - ")
        t = t.replace("\xa0", " ")

        # Fix hyphenated words broken across line wraps: e.g. "differen-\ntial" -> "differential"
        t = re.sub(r"(\b[a-zA-Z]+)-\n([a-zA-Z]+\b)", r"\1\2", t)

        # Remove repeated garbage characters from bad OCR (e.g. "------" or "____")
        t = re.sub(r"([_=\-~*]){5,}", r"\1\1\1", t)

        # Standardize multiple spaces within lines but preserve newlines
        t = re.sub(r"[ \t]+", " ", t)

        # Collapse excessive blank lines
        t = re.sub(r"\n{3,}", "\n\n", t)

        return t.strip()

    @staticmethod
    def assess_text_quality(text: str) -> Tuple[bool, int]:
        """
        Evaluates whether extracted text is usable digital text or requires OCR.
        Returns: (is_usable: bool, quality_score: int [0-100])
        """
        if not text or len(text.strip()) < 40:
            return False, 0

        clean = text.strip()
        total_chars = len(clean)
        alnum_chars = len(re.findall(r"[a-zA-Z0-9]", clean))
        ratio = alnum_chars / max(total_chars, 1)

        quality_score = min(100, max(0, int(ratio * 100)))
        # Text is considered usable if it has at least 45 chars and >= 60% alphanumeric ratio
        is_usable = (total_chars >= 45) and (ratio >= 0.60)
        return is_usable, quality_score

    def inspect_pdf(self, file_bytes: bytes, filename: str) -> Tuple[bool, Optional[str], int, str, int]:
        """
        Validate PDF: returns (is_valid, error_msg, total_pages, document_id, file_size_kb).
        """
        if pymupdf is None:
            return False, "PDF processing library (PyMuPDF) is not installed in the active environment.", 0, "", 0

        if not file_bytes or len(file_bytes) == 0:
            return False, "The uploaded file is empty.", 0, "", 0

        doc_id = self.calculate_document_id(file_bytes)
        file_size_kb = len(file_bytes) // 1024

        try:
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
            if doc.is_encrypted:
                doc.close()
                return False, "This PDF is password-protected. Please upload an unlocked PDF.", 0, doc_id, file_size_kb
            total_pages = len(doc)
            if total_pages == 0:
                doc.close()
                return False, "This PDF contains no readable pages.", 0, doc_id, file_size_kb
            doc.close()
            return True, None, total_pages, doc_id, file_size_kb
        except Exception as e:
            logger.warning(f"Error inspecting PDF {filename}: {e}")
            return False, "Could not open file as a valid PDF document.", 0, doc_id, file_size_kb

    def process_pdf(
        self,
        file_bytes: bytes,
        filename: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Two-stage extraction pipeline:
        - Step 1: PyMuPDF selectable text extraction + quality assessment
        - Step 2: High-resolution (300 DPI) Tesseract OCR with contrast enhancement for scanned pages
        - Strict page preservation with unique document_id
        """
        if pymupdf is None:
            raise RuntimeError("PyMuPDF is not installed in the active Python environment.")

        doc_id = self.calculate_document_id(file_bytes)
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        total_pages = len(doc)
        pages_data = []

        logger.info(f"Processing PDF '{filename}' [ID: {doc_id}] ({total_pages} pages)...")

        for idx in range(total_pages):
            page_num = idx + 1
            page = doc[idx]

            if progress_callback:
                progress_callback(page_num, total_pages, f"Reading page {page_num} of {total_pages}...")

            raw_text = page.get_text("text") or ""
            cleaned = self.clean_text(raw_text)
            is_usable, text_score = self.assess_text_quality(cleaned)

            if is_usable:
                source_type = "pdf_text"
                final_text = cleaned
                quality_score = text_score
            else:
                # Scanned or image page: run preprocessed Tesseract OCR
                source_type = "ocr"
                ocr_text = ""
                quality_score = 0

                if HAS_PYTESSERACT:
                    try:
                        # 1. Render at 300 DPI
                        pix = page.get_pixmap(dpi=self.dpi)
                        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")  # Grayscale

                        # 2. Increase contrast and sharpness
                        enhancer = ImageEnhance.Contrast(img)
                        enhanced = enhancer.enhance(1.9)
                        sharp = ImageEnhance.Sharpness(enhanced).enhance(1.4)

                        # 3. Tesseract OCR with page segmentation mode 6 (uniform block of text)
                        custom_config = r'--oem 3 --psm 6'
                        ocr_text = pytesseract.image_to_string(sharp, config=custom_config) or ""
                    except Exception as ocr_err:
                        logger.debug(f"Pytesseract error on page {page_num}: {ocr_err}")

                # MuPDF built-in OCR fallback
                if not ocr_text.strip() and hasattr(page, "get_textpage_ocr"):
                    try:
                        ocr_text = page.get_textpage_ocr().extractText() or ""
                    except Exception as mupdf_ocr_err:
                        logger.debug(f"MuPDF OCR error on page {page_num}: {mupdf_ocr_err}")

                final_text = self.clean_text(ocr_text) if ocr_text.strip() else cleaned
                if final_text:
                    _, quality_score = self.assess_text_quality(final_text)
                else:
                    source_type = "schematic_or_empty"
                    final_text = f"[Page {page_num} contains diagrams, charts, or visual borehole schematics with no textual records]"
                    quality_score = 15

            low_confidence = (quality_score < 40 and source_type == "ocr")

            pages_data.append({
                "document_id": doc_id,
                "filename": filename,
                "page_number": page_num,
                "text": final_text,
                "source_type": source_type,
                "quality_score": quality_score,
                "low_confidence": low_confidence,
                "char_count": len(final_text),
            })

        doc.close()
        logger.info(f"Finished processing '{filename}' [ID: {doc_id}] ({total_pages} pages).")
        return pages_data

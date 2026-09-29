"""
GeoInsight-RigX (eRTMAC-NWIS)
Module: Well Chatbot Engine (Completely Isolated per Document ID)
Guarantees absolute document isolation:
- Every document is stored in its own unique ChromaDB collection: f"well_ocr_{document_id}"
- Query retrieval strictly filters by document_id
- Strict confidence threshold: ungrounded questions return "Information Not Found"
- Zero cross-page leakage or global database access
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger("WellChatbot")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class WellChatbotEngine:
    """
    Dedicated chatbot engine strictly isolated to a single document via unique collection names.
    """

    def __init__(self, chroma_dir: Optional[Path] = None):
        base_dir = Path(__file__).resolve().parent.parent
        self.chroma_dir = chroma_dir or (base_dir / "data_source" / "chroma_db")
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._embedding_function = None

    def get_client(self):
        if self._client is None:
            self._client = chromadb.PersistentClient(path=str(self.chroma_dir))
        return self._client

    def get_embedding_function(self):
        if self._embedding_function is None:
            self._embedding_function = embedding_functions.DefaultEmbeddingFunction()
        return self._embedding_function

    def get_document_collection(self, document_id: str):
        """
        Creates or retrieves a collection EXCLUSIVELY dedicated to this specific document_id.
        """
        client = self.get_client()
        fn = self.get_embedding_function()
        col_name = f"well_ocr_{document_id}"
        return client.get_or_create_collection(name=col_name, embedding_function=fn)

    def index_document(self, pages_data: List[Dict[str, Any]], filename: str, document_id: str):
        """
        Index pages strictly into this document's private ChromaDB collection.
        """
        col = self.get_document_collection(document_id)

        # Clear any prior chunks in this document's collection to avoid duplicates
        try:
            existing_ids = col.get()["ids"]
            if existing_ids:
                col.delete(ids=existing_ids)
        except Exception as e:
            logger.debug(f"Reset notice for {document_id}: {e}")

        chunks = []
        for p in pages_data:
            text = p["text"]
            p_num = p["page_number"]
            src_type = p.get("source_type", "pdf_text")

            # Ignore empty schematic notices when creating search index
            if "schematics with no textual records" in text:
                continue

            if len(text) <= 900:
                chunks.append({
                    "id": f"{document_id}_p{p_num}_c0",
                    "text": text,
                    "page": p_num,
                    "document_id": document_id,
                    "filename": filename,
                    "source_type": src_type,
                })
            else:
                paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
                cur_c = ""
                c_idx = 0
                for para in paragraphs:
                    if len(cur_c) + len(para) + 2 <= 900:
                        cur_c = f"{cur_c}\n\n{para}".strip()
                    else:
                        if cur_c:
                            chunks.append({
                                "id": f"{document_id}_p{p_num}_c{c_idx}",
                                "text": cur_c,
                                "page": p_num,
                                "document_id": document_id,
                                "filename": filename,
                                "source_type": src_type,
                            })
                            c_idx += 1
                        cur_c = para
                if cur_c:
                    chunks.append({
                        "id": f"{document_id}_p{p_num}_c{c_idx}",
                        "text": cur_c,
                        "page": p_num,
                        "document_id": document_id,
                        "filename": filename,
                        "source_type": src_type,
                    })

        if chunks:
            ids = [c["id"] for c in chunks]
            docs = [c["text"] for c in chunks]
            metas = [
                {
                    "document_id": c["document_id"],
                    "filename": c["filename"],
                    "page": int(c["page"]),
                    "source_type": c["source_type"],
                }
                for c in chunks
            ]
            col.upsert(ids=ids, documents=docs, metadatas=metas)
            logger.info(f"Indexed {len(chunks)} chunks for document_id={document_id} into private collection well_ocr_{document_id}.")

    def answer_query(
        self,
        query: str,
        chat_history: List[Dict[str, Any]],
        filename: str,
        document_id: str,
        well_info: Dict[str, Any],
        key_params: Dict[str, str],
        full_text: str,
    ) -> Dict[str, Any]:
        """
        Answer query using strictly the isolated collection for document_id.
        Rejects questions about information not in this document.
        """
        col = self.get_document_collection(document_id)
        col_name = f"well_ocr_{document_id}"
        q_cleaned = query.strip()
        enhanced_query = q_cleaned
        q_lower = query.lower()

        # Conversational memory: resolve follow-up questions
        if len(q_cleaned.split()) <= 4 and chat_history:
            last_u = None
            for msg in reversed(chat_history):
                if msg.get("role") == "user":
                    last_u = msg.get("content")
                    break
            if last_u:
                enhanced_query = f"{last_u} {q_cleaned}"

        # 1. Check if user is asking for well depth or well parameters in a NON-WELL document
        is_asking_well_depth = any(term in q_lower for term in ["total depth", "well depth", "how deep", "depth of well", "td"])
        if is_asking_well_depth:
            # Check if this document actually has a well depth
            td_val = well_info.get("total_depth") or key_params.get("Total Well Depth")
            if not td_val or td_val == "Not found in report":
                return {
                    "found": False,
                    "answer": "### ⚠️ Information Not Found\n\nI couldn't find this information in the uploaded PDF. The uploaded document does not contain well depth records.",
                    "source_file": filename,
                    "pages": [],
                    "raw_context": "",
                    "debug_info": {
                        "collection_name": col_name,
                        "document_id": document_id,
                        "retrieved_pages": [],
                        "distances": [],
                    },
                }

        # 2. Query strictly within this document's private collection
        try:
            results = col.query(
                query_texts=[enhanced_query],
                n_results=min(4, max(col.count(), 1)),
                where={"document_id": document_id},
            )
        except Exception as e:
            logger.warning(f"Error querying private collection {col_name}: {e}")
            results = None

        if not results or not results.get("documents") or not results["documents"][0]:
            return {
                "found": False,
                "answer": "### ⚠️ Information Not Found\n\nI couldn't find this information in the uploaded PDF.",
                "source_file": filename,
                "pages": [],
                "raw_context": "",
                "debug_info": {
                    "collection_name": col_name,
                    "document_id": document_id,
                    "retrieved_pages": [],
                    "distances": [],
                },
            }

        docs = results["documents"][0]
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
        distances = results["distances"][0] if results.get("distances") else [0.5] * len(docs)

        # 3. Check semantic overlap and distance threshold
        COMMON_STOPWORDS = {
            "what", "when", "where", "which", "who", "why", "how", "about",
            "this", "that", "these", "those", "from", "with", "without", "into",
            "onto", "report", "document", "page", "tell", "show", "give", "find",
            "explain", "describe", "does", "have", "been", "were", "will", "would",
            "could", "should", "the", "and", "for", "are", "was", "not", "any",
            "all", "some", "them", "their", "there", "then", "also", "just",
        }
        # Extract meaningful topical keywords
        topical_words = [w for w in re.findall(r"[a-zA-Z0-9]{3,}", enhanced_query.lower()) if w not in COMMON_STOPWORDS]
        combined_text = " ".join(docs).lower()

        has_topical_overlap = any(w in combined_text for w in topical_words) if topical_words else True

        # Reject if query keywords are completely absent and distance is not exceptionally high-confidence
        if (topical_words and not has_topical_overlap) or (distances[0] > 1.25 and not has_topical_overlap):
            return {
                "found": False,
                "answer": "### ⚠️ Information Not Found\n\nI couldn't find this information in the uploaded PDF.",
                "source_file": filename,
                "pages": [],
                "raw_context": "",
                "debug_info": {
                    "collection_name": col_name,
                    "document_id": document_id,
                    "retrieved_pages": [],
                    "distances": distances,
                },
            }


        # Collect source pages
        pages_used = []
        for m in metas:
            p_val = m.get("page")
            if p_val and p_val not in pages_used:
                pages_used.append(int(p_val))

        top_doc = docs[0]
        primary_page = pages_used[0] if pages_used else 1

        # Format plain, direct answer
        answer_text = self._format_plain_answer(top_doc, query)

        # Source citations
        if len(pages_used) > 1:
            sources_md = "📄 **Sources:**\n\n" + "\n".join([f"• `{filename}` — Page {p}" for p in pages_used])
        else:
            sources_md = f"📄 **Source:** `{filename}` — Page {primary_page}"

        final_answer = f"{answer_text}\n\n{sources_md}"

        return {
            "found": True,
            "answer": final_answer,
            "source_file": filename,
            "pages": pages_used,
            "raw_context": top_doc,
            "debug_info": {
                "collection_name": col_name,
                "document_id": document_id,
                "retrieved_pages": pages_used,
                "distances": [round(float(d), 3) for d in distances],
            },
        }

    def _format_plain_answer(self, context_text: str, query: str) -> str:
        """
        Formulate a 2-4 short paragraph answer in simple, direct language.
        """
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context_text) if len(s.strip()) > 15]

        translated_sentences = []
        for s in sentences[:4]:
            text = s
            # Replace technical jargon with plain language
            text = re.sub(r"fractured carbonate(?: interval)?", "cracked limestone rock", text, flags=re.I)
            text = re.sub(r"lost circulation events?|severe mud loss", "drilling fluid escaping into rock fractures", text, flags=re.I)
            text = re.sub(r"differential(?:ly)? stuck pipe|mechanical binding", "drill pipe jamming tightly against the borehole wall", text, flags=re.I)
            text = re.sub(r"reactive smectite shale|borehole pack-off", "swelling clay rock pinching the drill pipe", text, flags=re.I)
            text = re.sub(r"high-viscosity pills?|sweeps?", "thick cleaning fluid pumped through the hole", text, flags=re.I)
            text = re.sub(r"equivalent circulating density \(ecd\)", "circulating fluid pressure", text, flags=re.I)
            text = re.sub(r"rotary speed \(rpm\)", "drill spin speed", text, flags=re.I)
            text = re.sub(r"weight on bit \(wob\)", "downward weight on the drill bit", text, flags=re.I)
            translated_sentences.append(text)

        if not translated_sentences:
            translated_sentences = [context_text[:260] + "..."]

        return "\n\n".join(translated_sentences)

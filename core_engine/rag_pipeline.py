"""
GeoInsight-RigX (eRTMAC-NWIS)
Core Engine - Open-Source RAG Extraction Pipeline
Extracts, chunks, and semantically indexes geological, wellbore, and hazard reports
from /data_source/reports using PyPDFLoader, RecursiveCharacterTextSplitter,
all-MiniLM-L6-v2 sentence-transformers, and ChromaDB vector store.
"""

import os
import re
import math
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

logger = logging.getLogger("RAGPipeline")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class SubsurfaceRAGPipeline:
    """
    Production-grade RAG pipeline for subsurface and geohazard technical literature.
    Ingests PDFs from data_source/reports, chunks them into 1000-char segments (200-char overlap),
    generates 384-dimensional dense vectors with all-MiniLM-L6-v2, and persists inside ChromaDB.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 200

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

        self._vector_store = None
        self._embedding_model = None

    def get_embedding_model(self):
        """
        Load or return the all-MiniLM-L6-v2 embedding model.
        Utilizes LangChain HuggingFaceEmbeddings or sentence_transformers.
        """
        if self._embedding_model is not None:
            return self._embedding_model

        logger.info(f"Loading embedding model: {self.MODEL_NAME}...")
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self._embedding_model = HuggingFaceEmbeddings(
                model_name=self.MODEL_NAME,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            return self._embedding_model
        except Exception as e:
            logger.warning(f"HuggingFaceEmbeddings init warning: {e}. Trying direct SentenceTransformer.")
            try:
                from sentence_transformers import SentenceTransformer
                raw_st = SentenceTransformer(self.MODEL_NAME)
                
                # Wrap to match LangChain embeddings interface
                class STLangChainWrapper:
                    def __init__(self, model):
                        self.model = model
                    def embed_documents(self, texts: List[str]) -> List[List[float]]:
                        return self.model.encode(texts, normalize_embeddings=True).tolist()
                    def embed_query(self, text: str) -> List[float]:
                        return self.model.encode([text], normalize_embeddings=True)[0].tolist()
                
                self._embedding_model = STLangChainWrapper(raw_st)
                return self._embedding_model
            except Exception as e2:
                logger.error(f"Could not load SentenceTransformer: {e2}")
                raise

    def load_and_chunk_pdfs(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> List[Any]:
        """
        Iterate through all PDF files located inside data_source/reports,
        extract text using PyPDFLoader, and split into chunks using RecursiveCharacterTextSplitter.
        """
        try:
            from langchain_community.document_loaders import PyPDFLoader
        except ImportError:
            from langchain.document_loaders import PyPDFLoader

        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

        pdf_files = list(self.reports_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.reports_dir}.")
            # Trigger report generator if empty
            try:
                from core_engine.generate_sample_reports import generate_all_reports
                generate_all_reports(str(self.reports_dir))
                pdf_files = list(self.reports_dir.glob("*.pdf"))
            except Exception as err:
                logger.error(f"Failed to auto-generate sample reports: {err}")

        all_docs = []
        for pdf_path in pdf_files:
            logger.info(f"Loading document via PyPDFLoader: {pdf_path.name}")
            try:
                loader = PyPDFLoader(str(pdf_path))
                docs = loader.load()
                # Ensure filename is clean in metadata
                for d in docs:
                    d.metadata["source_file"] = pdf_path.name
                    # 1-indexed page numbering
                    if "page" in d.metadata and isinstance(d.metadata["page"], int):
                        d.metadata["page_number"] = d.metadata["page"] + 1
                    else:
                        d.metadata["page_number"] = 1
                all_docs.extend(docs)
            except Exception as e:
                logger.error(f"Error loading {pdf_path.name} with PyPDFLoader: {e}")

        logger.info(f"Loaded {len(all_docs)} raw document pages from {len(pdf_files)} PDFs.")

        # Semantic chunking via RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
        )
        chunks = splitter.split_documents(all_docs)
        logger.info(f"Generated {len(chunks)} text chunks (chunk_size={chunk_size}, overlap={chunk_overlap}).")
        return chunks

    def build_or_load_vector_store(self, force_rebuild: bool = False) -> Any:
        """
        Embeds the PDF chunks and stores them persistently in a local Chroma directory,
        preventing the system from re-embedding reports on every Streamlit app reload.
        """
        if self._vector_store is not None and not force_rebuild:
            return self._vector_store

        try:
            from langchain_community.vectorstores import Chroma
        except ImportError:
            from langchain_chroma import Chroma

        embeddings = self.get_embedding_model()

        # Check if persistent vectorstore already exists on disk
        chroma_sqlite = self.chroma_dir / "chroma.sqlite3"
        has_existing_db = chroma_sqlite.is_file()

        if has_existing_db and not force_rebuild:
            logger.info(f"Loading existing Chroma vector database from: {self.chroma_dir}")
            try:
                self._vector_store = Chroma(
                    persist_directory=str(self.chroma_dir),
                    embedding_function=embeddings,
                )
                # Verify collection is non-empty
                count = self._vector_store._collection.count()
                if count > 0:
                    logger.info(f"ChromaDB ready with {count} indexed document embeddings.")
                    return self._vector_store
                else:
                    logger.warning("Existing Chroma collection is empty. Re-indexing...")
            except Exception as e:
                logger.warning(f"Failed to load existing Chroma index: {e}. Rebuilding...")

        # Build fresh vector store from PDFs
        chunks = self.load_and_chunk_pdfs()
        if not chunks:
            raise ValueError(f"No document chunks available to index in {self.reports_dir}")

        logger.info(f"Embedding {len(chunks)} chunks and persisting to ChromaDB at {self.chroma_dir}...")
        self._vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(self.chroma_dir),
        )
        logger.info("ChromaDB indexing completed and persistently saved to disk.")
        return self._vector_store

    def query_historical_reports(
        self,
        user_query: str,
        k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Perform a semantic similarity search against the ChromaDB vector store
        and return the top 3 most relevant paragraph chunks with metadata.
        
        Returns:
            List of dictionaries:
            - 'content': str
            - 'source_file': str
            - 'page_number': int
            - 'score': float (relevance score)
        """
        if not user_query or not user_query.strip():
            return []

        try:
            vectorstore = self.build_or_load_vector_store()
            
            # Use similarity search with score
            docs_and_scores = vectorstore.similarity_search_with_score(user_query, k=k)
            
            results = []
            for doc, score in docs_and_scores:
                meta = doc.metadata or {}
                source_file = meta.get("source_file") or meta.get("source", "Technical Report")
                # Format source file to base name
                source_name = Path(source_file).name
                page_num = meta.get("page_number") or (meta.get("page", 0) + 1)
                
                results.append({
                    "content": doc.page_content.strip(),
                    "source_file": source_name,
                    "page_number": int(page_num),
                    "score": round(float(score), 4),
                })
            
            logger.info(f"RAG query: '{user_query}' -> returned {len(results)} chunks.")
            return results

        except Exception as e:
            logger.error(f"Chroma similarity search encountered an error: {e}. Executing resilient semantic search.")
            return self._resilient_keyword_search(user_query, k=k)

    def _resilient_keyword_search(self, user_query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Fallback keyword / token overlap search in case of vector DB driver interruptions.
        """
        chunks = self.load_and_chunk_pdfs()
        query_words = set(re.findall(r"\w+", user_query.lower()))

        scored_chunks = []
        for c in chunks:
            text = c.page_content.lower()
            overlap = sum(1 for w in query_words if w in text)
            if overlap > 0:
                meta = c.metadata or {}
                source_name = Path(meta.get("source_file", meta.get("source", "Report.pdf"))).name
                page_num = meta.get("page_number", meta.get("page", 0) + 1)
                scored_chunks.append({
                    "content": c.page_content.strip(),
                    "source_file": source_name,
                    "page_number": int(page_num),
                    "score": round(float(overlap / max(1, len(query_words))), 3),
                })

        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:k]


# Global pipeline instance
_global_rag_pipeline: Optional[SubsurfaceRAGPipeline] = None

def get_rag_pipeline() -> SubsurfaceRAGPipeline:
    global _global_rag_pipeline
    if _global_rag_pipeline is None:
        _global_rag_pipeline = SubsurfaceRAGPipeline()
    return _global_rag_pipeline

def build_or_load_vector_store(force_rebuild: bool = False):
    return get_rag_pipeline().build_or_load_vector_store(force_rebuild=force_rebuild)

def query_historical_reports(user_query: str, k: int = 3) -> List[Dict[str, Any]]:
    return get_rag_pipeline().query_historical_reports(user_query=user_query, k=k)

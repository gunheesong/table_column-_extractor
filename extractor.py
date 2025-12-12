"""
Main API for table column extraction.

Pipeline:
1. Extract tables from PDFs (text content)
2. Use text embeddings to find top K relevant tables
3. Convert those pages to images
4. Send to Granite Vision VLM for column extraction

Note: Text embedding stays loaded (small). VLM is loaded on-demand.
Table embeddings are cached in ChromaDB for persistence across queries.
"""

import hashlib
from pathlib import Path

import chromadb
import torch

from .pdf_utils import extract_tables_from_pdf, page_to_image, get_all_pdfs, TableInfo
from .embeddings import TextEmbedder
from .vision_model import GraniteVisionExtractor
from .models import ExtractionResponse, PageResult


def _hash_pdf(path: str) -> str:
    """Get content hash of a PDF."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _clear_gpu_memory():
    """Free GPU memory after unloading a model."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class TableColumnExtractor:
    """
    Main API for extracting numeric column values from PDF tables.
    
    Example:
        extractor = TableColumnExtractor(
            text_embedding_path="/path/to/granite-embedding",
            granite_vision_path="/path/to/granite-vision-3.3-2b",
        )
        
        # Index PDFs once (embeddings cached in ChromaDB)
        extractor.index("/path/to/pdfs")
        
        # Query multiple times without recomputing embeddings
        result = extractor.extract("stress vs strain data", "strain")
        result2 = extractor.extract("temperature readings", "temp")
    """
    
    def __init__(
        self,
        text_embedding_path: str,
        granite_vision_path: str,
        chroma_path: str = "./chroma_db",
    ):
        """
        Initialize the extractor with model paths.
        
        Text embedding model is loaded immediately (small, ~1-2GB).
        VLM is loaded on-demand during extract().
        
        Args:
            text_embedding_path: Path to Granite text embedding model
            granite_vision_path: Path to Granite Vision 3.3 2B model (VLM)
            chroma_path: Path to ChromaDB storage (for caching embeddings)
        """
        # Load text embedder once (small model, keep in memory)
        print("Loading text embedding model...")
        self.text_embedder = TextEmbedder(text_embedding_path)
        print("Text embedding model loaded.")
        
        # ChromaDB for caching table embeddings
        self.chroma = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma.get_or_create_collection("tables")
        
        # Store path for VLM (loaded on-demand)
        self.granite_vision_path = granite_vision_path
    
    def index(self, pdf_directory: str) -> int:
        """
        Index PDFs into ChromaDB. Call this once when PDFs are uploaded.
        Skips already-indexed files.
        
        Returns:
            Number of newly indexed PDFs
        """
        pdf_files = get_all_pdfs(pdf_directory)
        indexed = 0
        
        for pdf_path in pdf_files:
            pdf_hash = _hash_pdf(str(pdf_path))
            
            # Skip if already indexed
            existing = self.collection.get(where={"pdf_hash": pdf_hash})
            if existing["ids"]:
                continue
            
            try:
                tables = extract_tables_from_pdf(pdf_path)
                if not tables:
                    continue
                
                texts = [t.text_content for t in tables]
                embeddings = self.text_embedder.embed(texts).cpu().tolist()
                
                self.collection.add(
                    ids=[f"{pdf_hash}_{i}" for i in range(len(tables))],
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=[{
                        "pdf_path": str(t.pdf_path),
                        "pdf_name": Path(t.pdf_path).name,
                        "page_num": t.page_num,
                        "pdf_hash": pdf_hash,
                    } for t in tables],
                )
                indexed += 1
                print(f"Indexed {pdf_path.name}: {len(tables)} tables")
            except Exception as e:
                print(f"Warning: Failed to index {pdf_path}: {e}")
        
        return indexed
    
    def extract(
        self,
        table_description: str,
        column_name: str,
        top_k: int = 5,
        dpi: int = 150,
    ) -> ExtractionResponse:
        """
        Extract numeric values from a specified column in matching tables.
        
        Must call index() first to populate the cache.
        
        Args:
            table_description: Description of the table to find 
                               (e.g., "stress vs strain data", "temperature readings")
            column_name: Name of the column to extract values from
                         (e.g., "strain", "y", "temperature")
            top_k: Number of candidate pages to send to VLM (default 5)
            dpi: Resolution for PDF page rendering
            
        Returns:
            ExtractionResponse with extracted values and source info
        """
        # Query ChromaDB for top tables (must call index() first)
        query_embedding = self.text_embedder.embed_single(table_description).cpu().tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas"],
        )
        
        if not results["ids"] or not results["ids"][0]:
            return ExtractionResponse(
                table_description=table_description,
                column_name=column_name,
                results=[],
            )
        
        top_tables = results["metadatas"][0]  # List of metadata dicts
        
        # Step 3: Convert pages to images (deduplicate by page)
        page_keys = set()
        unique_pages = []
        for meta in top_tables:
            key = (meta["pdf_path"], meta["page_num"])
            if key not in page_keys:
                page_keys.add(key)
                unique_pages.append(meta)
        
        # Convert pages to images
        images = []
        page_info = []
        for meta in unique_pages:
            img = page_to_image(meta["pdf_path"], meta["page_num"], dpi=dpi)
            images.append(img)
            page_info.append({
                "pdf": meta["pdf_name"],
                "page": meta["page_num"],
            })
        
        # Step 4: Load VLM, extract column values (one image at a time), then unload
        print("Loading Granite Vision VLM...")
        vision_extractor = GraniteVisionExtractor(self.granite_vision_path)
        
        # Process each image separately (constant memory usage)
        extraction_results = vision_extractor.extract_column_values(
            images,
            table_description=table_description,
            column_name=column_name,
        )
        
        # Unload VLM
        del vision_extractor
        _clear_gpu_memory()
        print("Granite Vision VLM unloaded.")
        
        # Build per-page results
        results = []
        for info, col_values in zip(page_info, extraction_results):
            results.append(PageResult(
                pdf=info["pdf"],
                page=info["page"],
                values=col_values.values,
            ))
        
        return ExtractionResponse(
            table_description=table_description,
            column_name=column_name,
            results=results,
        )

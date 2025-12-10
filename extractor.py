"""
Main API for table column extraction.

Pipeline:
1. Extract tables from PDFs (text content)
2. Use text embeddings to find top K relevant tables
3. Convert those pages to images
4. Send to Granite Vision VLM for column extraction

Note: Text embedding stays loaded (small). VLM is loaded on-demand.
"""

from pathlib import Path

import torch

from .pdf_utils import extract_tables_from_pdf, page_to_image, get_all_pdfs, TableInfo
from .embeddings import TextEmbedder, compute_similarity
from .vision_model import GraniteVisionExtractor
from .models import ExtractionResponse, PageResult


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
        
        result = extractor.extract(
            table_description="stress vs strain data",
            column_name="strain",
            pdf_directory="/path/to/pdfs",
        )
        print(result.values)  # [100, 200, 300, ...]
    """
    
    def __init__(
        self,
        text_embedding_path: str,
        granite_vision_path: str,
    ):
        """
        Initialize the extractor with model paths.
        
        Text embedding model is loaded immediately (small, ~1-2GB).
        VLM is loaded on-demand during extract().
        
        Args:
            text_embedding_path: Path to Granite text embedding model
            granite_vision_path: Path to Granite Vision 3.3 2B model (VLM)
        """
        # Load text embedder once (small model, keep in memory)
        print("Loading text embedding model...")
        self.text_embedder = TextEmbedder(text_embedding_path)
        print("Text embedding model loaded.")
        
        # Store path for VLM (loaded on-demand)
        self.granite_vision_path = granite_vision_path
    
    def extract(
        self,
        table_description: str,
        column_name: str,
        pdf_directory: str,
        top_k: int = 5,
        dpi: int = 150,
    ) -> ExtractionResponse:
        """
        Extract numeric values from a specified column in matching tables.
        
        Args:
            table_description: Description of the table to find 
                               (e.g., "stress vs strain data", "temperature readings")
            column_name: Name of the column to extract values from
                         (e.g., "strain", "y", "temperature")
            pdf_directory: Directory containing PDF files
            top_k: Number of candidate pages to send to VLM (default 5)
            dpi: Resolution for PDF page rendering
            
        Returns:
            ExtractionResponse with extracted values and source info
        """
        # Step 1: Extract all tables from PDFs (no model needed)
        pdf_files = get_all_pdfs(pdf_directory)
        all_tables: list[TableInfo] = []
        
        for pdf_path in pdf_files:
            tables = extract_tables_from_pdf(pdf_path)
            all_tables.extend(tables)
        
        if not all_tables:
            return ExtractionResponse(
                table_description=table_description,
                column_name=column_name,
                values=[],
                source_pages=[],
            )
        
        # Step 2: Use text embedder to find top tables (already loaded in __init__)
        table_texts = [t.text_content for t in all_tables]
        query_embedding = self.text_embedder.embed_single(table_description)
        table_embeddings = self.text_embedder.embed(table_texts)
        
        text_similarities = compute_similarity(query_embedding, table_embeddings)
        top_indices = text_similarities.argsort(descending=True)[:top_k].tolist()
        top_tables = [all_tables[i] for i in top_indices]
        
        # Step 3: Convert pages to images (deduplicate by page)
        page_keys = set()
        unique_pages = []
        for table in top_tables:
            key = (table.pdf_path, table.page_num)
            if key not in page_keys:
                page_keys.add(key)
                unique_pages.append(table)
        
        images = []
        page_info = []
        for table in unique_pages:
            img = page_to_image(table.pdf_path, table.page_num, dpi=dpi)
            images.append(img)
            page_info.append({
                "pdf": Path(table.pdf_path).name,
                "page": table.page_num,
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

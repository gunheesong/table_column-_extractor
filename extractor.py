"""
Main API for table column extraction.

Pipeline:
1. Extract tables from PDFs (text content)
2. Use text embeddings to find top 5 relevant tables
3. Convert those pages to images
4. Use vision embeddings to select top 3 images
5. Send to Granite Vision for column extraction
"""

from pathlib import Path

from .pdf_utils import extract_tables_from_pdf, page_to_image, get_all_pdfs, TableInfo
from .embeddings import TextEmbedder, VisionEmbedder, compute_similarity
from .vision_model import GraniteVisionExtractor
from .models import ColumnValues, ExtractionResponse


class TableColumnExtractor:
    """
    Main API for extracting numeric column values from PDF tables.
    
    Example:
        extractor = TableColumnExtractor(
            text_embedding_path="/path/to/granite-embedding",
            vision_embedding_path="/path/to/clip-model",
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
        vision_embedding_path: str,
        granite_vision_path: str,
    ):
        """
        Initialize the extractor with model paths.
        
        Args:
            text_embedding_path: Path to Granite text embedding model
            vision_embedding_path: Path to vision embedding model (CLIP/SigLIP)
            granite_vision_path: Path to Granite Vision 3.3 2B model
        """
        self.text_embedder = TextEmbedder(text_embedding_path)
        self.vision_embedder = VisionEmbedder(vision_embedding_path)
        self.vision_extractor = GraniteVisionExtractor(granite_vision_path)
    
    def extract(
        self,
        table_description: str,
        column_name: str,
        pdf_directory: str,
        top_k_text: int = 5,
        top_k_vision: int = 3,
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
            top_k_text: Number of tables to select via text similarity (default 5)
            top_k_vision: Number of images to select via vision similarity (default 3)
            dpi: Resolution for PDF page rendering
            
        Returns:
            ExtractionResponse with extracted values and source info
        """
        # Step 1: Extract all tables from PDFs
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
        
        # Step 2: Embed table description and table texts, find top 5
        # Use table_description for semantic search (finding the right tables)
        table_texts = [t.text_content for t in all_tables]
        query_embedding = self.text_embedder.embed_single(table_description)
        table_embeddings = self.text_embedder.embed(table_texts)
        
        text_similarities = compute_similarity(query_embedding, table_embeddings)
        top_text_indices = text_similarities.argsort(descending=True)[:top_k_text].tolist()
        top_tables = [all_tables[i] for i in top_text_indices]
        
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
        
        # Step 4: Use vision embeddings to select top 3
        # Use table_description for vision similarity too
        if len(images) > top_k_vision:
            image_embeddings = self.vision_embedder.embed_images(images)
            query_vision_embedding = self.vision_embedder.embed_text(table_description)
            
            vision_similarities = compute_similarity(query_vision_embedding, image_embeddings)
            top_vision_indices = vision_similarities.argsort(descending=True)[:top_k_vision].tolist()
            
            selected_images = [images[i] for i in top_vision_indices]
            selected_page_info = [page_info[i] for i in top_vision_indices]
        else:
            selected_images = images
            selected_page_info = page_info
        
        # Step 5: Extract column values with Granite Vision
        # Pass both table_description and column_name for accurate extraction
        result = self.vision_extractor.extract_column_values(
            selected_images,
            table_description=table_description,
            column_name=column_name,
        )
        
        return ExtractionResponse(
            table_description=table_description,
            column_name=column_name,
            values=result.values,
            source_pages=selected_page_info,
        )

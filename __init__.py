"""
Table Column Extractor

A pipeline for extracting numeric column data from PDF tables using:
- Granite text embeddings for semantic table search
- Vision embeddings for image-based re-ranking
- Granite Vision 3.3 2B for structured data extraction
"""

from .extractor import TableColumnExtractor
from .models import ColumnValues, ExtractionRequest, ExtractionResponse

__all__ = [
    "TableColumnExtractor",
    "ColumnValues",
    "ExtractionRequest",
    "ExtractionResponse",
]


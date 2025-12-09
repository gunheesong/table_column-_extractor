"""
Pydantic models for structured input/output.
"""

from pydantic import BaseModel


class ColumnValues(BaseModel):
    """Structured output: list of integers extracted from a column."""
    values: list[int]


class ExtractionRequest(BaseModel):
    """Request to extract column values from PDFs."""
    query: str  # Must specify the column name to extract
    pdf_directory: str  # Directory containing PDF files
    

class ExtractionResponse(BaseModel):
    """Response containing extracted column values."""
    query: str
    values: list[int]
    source_pages: list[dict]  # Info about which pages the values came from


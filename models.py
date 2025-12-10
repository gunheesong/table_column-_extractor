"""
Pydantic models for structured input/output.
"""

from pydantic import BaseModel


class ColumnValues(BaseModel):
    """Structured output: list of integers extracted from a column."""
    values: list[int]


class PageResult(BaseModel):
    """Result from a single page."""
    pdf: str
    page: int
    values: list[int]


class ExtractionRequest(BaseModel):
    """Request to extract column values from PDFs."""
    table_description: str  # Description of the table to find (e.g., "stress vs strain data")
    column_name: str        # Column to extract (e.g., "strain", "y")
    pdf_directory: str      # Directory containing PDF files
    

class ExtractionResponse(BaseModel):
    """Response containing extracted column values per page."""
    table_description: str
    column_name: str
    results: list[PageResult]  # Results per page, displayed separately

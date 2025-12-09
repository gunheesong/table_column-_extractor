"""
PDF processing utilities.

Handles:
- Table detection and text extraction from PDFs
- Converting PDF pages to images
"""

from pathlib import Path
from dataclasses import dataclass

import fitz  # PyMuPDF
from PIL import Image


@dataclass
class TableInfo:
    """Information about a detected table."""
    pdf_path: str
    page_num: int
    bbox: tuple[float, float, float, float]
    text_content: str  # Markdown or text representation of table


def extract_tables_from_pdf(pdf_path: str | Path) -> list[TableInfo]:
    """
    Extract all tables from a PDF with their text content.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of TableInfo objects
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    tables = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        try:
            table_finder = page.find_tables()
            
            for table in table_finder.tables:
                # Extract text content as markdown
                text_content = _table_to_markdown(table)
                
                if text_content.strip():
                    tables.append(TableInfo(
                        pdf_path=str(pdf_path),
                        page_num=page_num,
                        bbox=table.bbox,
                        text_content=text_content,
                    ))
        except Exception:
            continue
    
    doc.close()
    return tables


def _table_to_markdown(table) -> str:
    """Convert PyMuPDF table to markdown string."""
    try:
        df = table.to_pandas()
        if df is not None and not df.empty:
            return df.to_markdown(index=False)
    except Exception:
        pass
    
    # Fallback: manually build markdown
    try:
        rows = []
        for row in table.extract():
            if row:
                row_texts = [str(cell) if cell else "" for cell in row]
                rows.append("| " + " | ".join(row_texts) + " |")
        
        if len(rows) > 1:
            # Insert separator after header
            separator = "| " + " | ".join(["---"] * len(rows[0].split("|")[1:-1])) + " |"
            rows.insert(1, separator)
        
        return "\n".join(rows)
    except Exception:
        return ""


def page_to_image(pdf_path: str | Path, page_num: int, dpi: int = 150) -> Image.Image:
    """
    Convert a PDF page to a PIL Image.
    
    Args:
        pdf_path: Path to the PDF file
        page_num: Page number (0-indexed)
        dpi: Resolution for rendering
        
    Returns:
        PIL Image of the page
    """
    doc = fitz.open(str(pdf_path))
    page = doc[page_num]
    pix = page.get_pixmap(dpi=dpi)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return image


def get_all_pdfs(directory: str | Path) -> list[Path]:
    """Get all PDF files in a directory (recursive)."""
    directory = Path(directory)
    return list(directory.glob("*.pdf")) + list(directory.glob("**/*.pdf"))


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
            # Get text blocks for context extraction
            text_blocks = page.get_text("dict")["blocks"]
            
            table_finder = page.find_tables()
            
            for table in table_finder.tables:
                table_bbox = table.bbox
                
                # Get context above the table (caption + section heading)
                context = _get_text_above_table(text_blocks, table_bbox)
                
                # Extract table content as markdown
                table_markdown = _table_to_markdown(table)
                
                # Combine: context + table
                if context:
                    text_content = f"{context}\n{table_markdown}"
                else:
                    text_content = table_markdown
                
                if text_content.strip():
                    tables.append(TableInfo(
                        pdf_path=str(pdf_path),
                        page_num=page_num,
                        bbox=table_bbox,
                        text_content=text_content,
                    ))
        except Exception:
            continue
    
    doc.close()
    return tables


def _get_text_above_table(text_blocks: list, table_bbox: tuple, max_distance: float = 100) -> str:
    """
    Get text above the table (likely caption and/or section heading).
    
    Args:
        text_blocks: List of text blocks from page.get_text("dict")
        table_bbox: Table bounding box (x0, y0, x1, y1)
        max_distance: Maximum pixels above table to search
        
    Returns:
        Combined text found above the table
    """
    table_top = table_bbox[1]
    
    # Collect text blocks above the table
    above_texts = []
    
    for block in text_blocks:
        if block.get("type") != 0:  # Skip non-text blocks
            continue
        
        block_bbox = block.get("bbox", (0, 0, 0, 0))
        block_bottom = block_bbox[3]
        
        # Check if block is above table and within max_distance
        if block_bottom <= table_top and (table_top - block_bottom) < max_distance:
            # Extract text from block
            text = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text += span.get("text", "") + " "
            
            text = text.strip()
            if text:
                above_texts.append((block_bottom, text))  # Store with y-position
    
    # Sort by y-position (top to bottom) and combine
    above_texts.sort(key=lambda x: x[0])
    return "\n".join([t[1] for t in above_texts])


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


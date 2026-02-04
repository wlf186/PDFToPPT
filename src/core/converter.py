"""Core converter module for PDF to PPT conversion."""

import io
from pathlib import Path
from typing import Optional

import pymupdf

from .pdf_parser import PDFElementExtractor
from .ppt_builder import PPTBuilder


# Conversion ratio: 1 inch = 914400 EMU, 1 inch = 72 points
PDF_TO_EMU = 12700


class PDFToPPTConverter:
    """Convert PDF files to editable PowerPoint presentations."""

    def __init__(
        self,
        pdf_path: str | Path,
        ppt_path: str | Path,
        use_ocr: bool = False,
        ocr_lang: str = "chi_sim+eng",
    ):
        """
        Initialize the converter.

        Args:
            pdf_path: Path to the input PDF file
            ppt_path: Path to the output PPTX file
            use_ocr: Whether to use OCR for text extraction
            ocr_lang: OCR language(s) to use
        """
        self.pdf_path = Path(pdf_path)
        self.ppt_path = Path(ppt_path)
        self.use_ocr = use_ocr
        self.ocr_lang = ocr_lang
        self.doc: Optional[pymupdf.Document] = None

    def convert(self) -> Path:
        """
        Perform the PDF to PPT conversion.

        Returns:
            Path to the created PPTX file
        """
        self.doc = pymupdf.open(self.pdf_path)

        # Get PDF page dimensions
        first_page = self.doc[0]
        pdf_width = first_page.rect.width
        pdf_height = first_page.rect.height

        # Initialize PPT builder with PDF dimensions
        builder = PPTBuilder(pdf_width, pdf_height)

        # Initialize PDF element extractor
        extractor = PDFElementExtractor(use_ocr=self.use_ocr, ocr_lang=self.ocr_lang)

        # Process each page
        for page_num, page in enumerate(self.doc):
            # Create a new slide for this page
            builder.add_slide()

            # Extract text blocks
            text_blocks = extractor.extract_text_blocks(page)
            for text_block in text_blocks:
                builder.add_text_box(
                    text=text_block.text,
                    bbox=text_block.bbox,
                    font_info=text_block.font_info,
                )

            # Extract images
            images = extractor.extract_images(page)
            for image_block in images:
                builder.add_image(
                    image_bytes=image_block.image_bytes,
                    bbox=image_block.bbox,
                )

            # Detect and add tables
            tables = extractor.detect_tables(page, text_blocks)
            for table_block in tables:
                builder.add_table(
                    rows=table_block.rows,
                    cols=table_block.cols,
                    bbox=table_block.bbox,
                    data=table_block.data,
                )

            # Check if page has any extractable content
            # If no text, images, or tables were found, render the whole page as an image
            # (handles scanned PDFs when OCR is not available)
            if not text_blocks and not images and not tables:
                rect = page.rect
                page_image = extractor.render_page_as_image(page)
                builder.add_image(
                    image_bytes=page_image,
                    bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                )

        self.doc.close()

        # Save the presentation
        builder.save(self.ppt_path)

        return self.ppt_path

    def get_page_count(self) -> int:
        """Get the number of pages in the PDF."""
        if self.doc is None:
            self.doc = pymupdf.open(self.pdf_path)
            count = len(self.doc)
            self.doc.close()
            return count
        return len(self.doc)

    @staticmethod
    def get_pdf_info(pdf_path: str | Path) -> dict:
        """
        Get information about a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary with PDF metadata
        """
        doc = pymupdf.open(pdf_path)
        first_page = doc[0]

        info = {
            "page_count": len(doc),
            "width": first_page.rect.width,
            "height": first_page.rect.height,
            "metadata": doc.metadata,
        }

        doc.close()
        return info

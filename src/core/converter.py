"""Core converter module for PDF to PPT conversion."""

import io
import logging
from pathlib import Path
from typing import Optional

import pymupdf

from .config import get_config
from .llm_client import LLMClient
from .pdf_parser import PDFElementExtractor
from .ppt_builder import PPTBuilder

logger = logging.getLogger(__name__)

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
        use_llm: bool = True,
    ):
        """
        Initialize the converter.

        Args:
            pdf_path: Path to the input PDF file
            ppt_path: Path to the output PPTX file
            use_ocr: Whether to use OCR for text extraction
            ocr_lang: OCR language(s) to use
            use_llm: Whether to use LLM enhancement (if configured)
        """
        self.pdf_path = Path(pdf_path)
        self.ppt_path = Path(ppt_path)
        self.use_ocr = use_ocr
        self.ocr_lang = ocr_lang
        self.use_llm = use_llm
        self.doc: Optional[pymupdf.Document] = None

        # Load LLM configuration
        self.config = get_config()
        self.llm_client: Optional[LLMClient] = None

        if self.use_llm and self.config.llm.enabled:
            self.llm_client = LLMClient(self.config.llm)

    def get_llm_status(self) -> dict:
        """
        Get LLM status information.

        Returns:
            Dictionary with LLM status
        """
        if not self.use_llm or not self.config.llm.enabled:
            return {
                "enabled": False,
                "configured": False,
                "message": "LLM enhancement is disabled",
            }

        if self.llm_client is None:
            return {
                "enabled": True,
                "configured": False,
                "message": "LLM client not initialized",
            }

        if self.llm_client.is_available():
            return {
                "enabled": True,
                "configured": True,
                "available": True,
                "model": self.config.llm.model_name,
                "base_url": self.config.llm.base_url,
                "message": f"Using {self.config.llm.model_name} @ {self.config.llm.base_url}",
            }
        else:
            return {
                "enabled": True,
                "configured": self.config.llm.is_configured(),
                "available": False,
                "error": self.llm_client.get_error(),
                "message": f"LLM not available: {self.llm_client.get_error()}",
            }

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

        # Check LLM status
        llm_available = False
        if self.use_llm and self.llm_client and self.llm_client.is_available():
            llm_available = True
            logger.info(f"LLM enhancement enabled: {self.llm_client.get_info()}")

        # Process each page
        for page_num, page in enumerate(self.doc):
            # Create a new slide for this page
            builder.add_slide()

            # Collect all page text for LLM context
            page_text = ""

            # Extract text blocks
            text_blocks = extractor.extract_text_blocks(page)
            for text_block in text_blocks:
                page_text += text_block.text + "\n"
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

            # Use LLM for enhancement if available
            if llm_available and self.llm_client:
                try:
                    # Render page as image for LLM
                    page_image = extractor.render_page_as_image(page)

                    # Get LLM enhancement
                    enhancement = self.llm_client.enhance_page_sync(
                        page_image=page_image,
                        page_number=page_num + 1,
                        extracted_text=page_text,
                    )

                    if "error" not in enhancement:
                        logger.debug(f"Page {page_num + 1} LLM enhancement successful")
                        # Enhancement data available for future use
                        # Can be used to improve layout, styling, etc.
                    else:
                        logger.warning(f"Page {page_num + 1} LLM enhancement failed: {enhancement.get('error')}")

                except Exception as e:
                    logger.warning(f"LLM enhancement failed for page {page_num + 1}: {e}")

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

    @staticmethod
    def get_global_llm_status() -> dict:
        """
        Get global LLM configuration status (without initializing converter).

        Returns:
            Dictionary with LLM configuration status
        """
        config = get_config()

        if not config.llm.enabled:
            return {
                "enabled": False,
                "configured": config.llm.is_configured(),
                "message": "LLM enhancement is disabled in config.yaml",
            }

        if not config.llm.is_configured():
            return {
                "enabled": True,
                "configured": False,
                "message": "LLM enabled but not configured. Set base_url and model_name in config.yaml",
            }

        # Try to initialize client and test connection
        client = LLMClient(config.llm)

        if not client.is_available():
            return {
                "enabled": True,
                "configured": True,
                "available": False,
                "error": client.get_error(),
                "message": f"LLM configured but not available: {client.get_error()}",
            }

        # Test connection
        success, message = client.test_connection_sync()
        return {
            "enabled": True,
            "configured": True,
            "available": success,
            "model": config.llm.model_name,
            "base_url": config.llm.base_url,
            "message": message,
        }

"""Core converter module for PDF to PPT conversion."""

import io
import logging
from pathlib import Path
from typing import Callable, Optional

import pymupdf

from .config import get_config
from .llm_client import LLMClient
from .pdf_parser import FontInfo, PDFElementExtractor
from .ppt_builder import PPTBuilder

logger = logging.getLogger(__name__)

# Conversion ratio: 1 inch = 914400 EMU, 1 inch = 72 points
PDF_TO_EMU = 12700

# Progress callback type: (current_page, total_pages, using_llm, message)
ProgressCallback = Callable[[int, int, bool, str], None]


class PDFToPPTConverter:
    """Convert PDF files to editable PowerPoint presentations."""

    def __init__(
        self,
        pdf_path: str | Path,
        ppt_path: str | Path,
        use_ocr: bool = False,
        ocr_lang: str = "chi_sim+eng",
        use_llm: bool = True,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        """
        Initialize the converter.

        Args:
            pdf_path: Path to the input PDF file
            ppt_path: Path to the output PPTX file
            use_ocr: Whether to use OCR for text extraction
            ocr_lang: OCR language(s) to use
            use_llm: Whether to use LLM enhancement (if configured)
            progress_callback: Optional callback for progress updates
        """
        self.pdf_path = Path(pdf_path)
        self.ppt_path = Path(ppt_path)
        self.use_ocr = use_ocr
        self.ocr_lang = ocr_lang
        self.use_llm = use_llm
        self.progress_callback = progress_callback
        self.doc: Optional[pymupdf.Document] = None

        # Load LLM configuration
        self.config = get_config()
        self.llm_client: Optional[LLMClient] = None

        if self.use_llm and self.config.llm.enabled:
            self.llm_client = LLMClient(self.config.llm)

    def _update_progress(self, current: int, total: int, using_llm: bool, message: str = ""):
        """Call progress callback if available."""
        if self.progress_callback:
            self.progress_callback(current, total, using_llm, message)

    def _apply_llm_enhancement(
        self,
        builder: PPTBuilder,
        page: pymupdf.Page,
        enhancement: dict,
        pdf_width: float,
        pdf_height: float,
    ) -> bool:
        """
        Apply LLM enhancement results to the slide.

        Args:
            builder: PPT builder instance
            page: Current PDF page
            enhancement: LLM enhancement result
            pdf_width: Page width in points
            pdf_height: Page height in points

        Returns:
            True if enhancement was successfully applied
        """
        try:
            content_blocks = enhancement.get("content_blocks", [])
            if not content_blocks:
                logger.warning("LLM returned no content blocks")
                return False

            # Clear the slide first (remove default text boxes)
            # Actually, we should add content AFTER clearing, but python-pptx doesn't support deletion easily
            # So we'll just add new text boxes on top

            for block in content_blocks:
                block_type = block.get("type", "paragraph")
                text = block.get("text", "").strip()
                if not text:
                    continue

                # Parse position (percentages to points)
                position = block.get("position", {})
                x0_pct = position.get("x0_pct", 0.0)
                y0_pct = position.get("y0_pct", 0.0)
                x1_pct = position.get("x1_pct", 1.0)
                y1_pct = position.get("y1_pct", 0.1)

                # Convert to points
                x0 = x0_pct * pdf_width
                y0 = y0_pct * pdf_height
                x1 = x1_pct * pdf_width
                y1 = y1_pct * pdf_height

                # Parse style
                style = block.get("style", {})
                font_size = style.get("font_size", 12)
                bold = style.get("bold", False)
                color = style.get("color", [0, 0, 0])

                # Create font info
                font_info = FontInfo(
                    font="Calibri",
                    size=font_size,
                    color=tuple(color) if isinstance(color, list) else (0, 0, 0),
                    bold=bold,
                    italic=False,
                )

                # Add text box to slide
                builder.add_text_box(
                    text=text,
                    bbox=(x0, y0, x1, y1),
                    font_info=font_info,
                )

            logger.info(f"Applied {len(content_blocks)} content blocks from LLM")
            return True

        except Exception as e:
            logger.error(f"Error applying LLM enhancement: {e}")
            return False

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
        total_pages = len(self.doc)

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
            logger.info(f"LLM available: {self.llm_client.get_info()}")
            if self.config.llm.use_enhancement:
                logger.info("LLM layout enhancement: ENABLED (may be slow on CPU)")
            else:
                logger.info("LLM layout enhancement: DISABLED (OCR only)")

        # Process each page
        for page_num, page in enumerate(self.doc):
            current_page = page_num + 1
            using_llm_this_page = False
            llm_enhancement_applied = False

            # Update progress - starting page
            self._update_progress(current_page, total_pages, False, f"Processing page {current_page}/{total_pages}")

            # Create a new slide for this page
            builder.add_slide()

            # Get page dimensions for this specific page
            page_rect = page.rect
            page_width = page_rect.width
            page_height = page_rect.height

            # First, try LLM enhancement if enabled (this will handle scanned PDFs via vision)
            if llm_available and self.llm_client and self.config.llm.use_enhancement:
                using_llm_this_page = True
                self._update_progress(current_page, total_pages, True, f"Page {current_page}: LLM analyzing...")

                try:
                    # Render page as image for LLM
                    llm_dpi = 100 if self.config.llm.preset == "ollama" else 150
                    page_image = extractor.render_page_as_image(page, dpi=llm_dpi)

                    logger.debug(f"LLM enhancement image: {len(page_image) / 1024:.1f} KB at {llm_dpi} DPI")

                    # Get LLM enhancement
                    enhancement = self.llm_client.enhance_page_sync(
                        page_image=page_image,
                        page_number=current_page,
                        extracted_text="",  # Let LLM extract everything from image
                    )

                    if "error" not in enhancement:
                        # Apply LLM enhancement to slide
                        if self._apply_llm_enhancement(builder, page, enhancement, page_width, page_height):
                            llm_enhancement_applied = True
                            logger.info(f"Page {current_page}: LLM enhancement applied successfully")
                        else:
                            logger.warning(f"Page {current_page}: LLM enhancement returned no usable data")
                            using_llm_this_page = False
                    else:
                        logger.warning(f"Page {current_page}: LLM enhancement failed: {enhancement.get('error')}")
                        using_llm_this_page = False

                except Exception as e:
                    logger.warning(f"LLM enhancement failed for page {current_page}: {e}")
                    using_llm_this_page = False

            # If LLM enhancement was not applied, fall back to traditional extraction
            if not llm_enhancement_applied:
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
                if not text_blocks and not images and not tables:
                    page_image = extractor.render_page_as_image(page)
                    builder.add_image(
                        image_bytes=page_image,
                        bbox=(page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y1),
                    )

            # Update progress - page completed
            llm_status = " [LLM]" if using_llm_this_page else ""
            self._update_progress(current_page, total_pages, using_llm_this_page, f"Page {current_page}/{total_pages} done{llm_status}")

        self.doc.close()

        # Update progress - saving
        self._update_progress(total_pages, total_pages, False, "Saving presentation...")

        # Save the presentation
        builder.save(self.ppt_path)

        # Update progress - complete
        self._update_progress(total_pages, total_pages, False, f"Saved to: {self.ppt_path}")

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

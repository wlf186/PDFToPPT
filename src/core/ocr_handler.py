"""OCR text extraction module for scanned PDFs."""

import io
from typing import List, Optional

import pymupdf
from PIL import Image


class OCRHandler:
    """Handle OCR text extraction using Tesseract."""

    def __init__(self, lang: str = "chi_sim+eng"):
        """
        Initialize the OCR handler.

        Args:
            lang: OCR language(s) to use (default: chi_sim+eng for Chinese and English)

        Note:
            If tesseract is not available, the handler will be in a disabled state.
            Check the `available` property to see if OCR is functional.
        """
        self.lang = lang
        self._tesseract = None
        self.available = False
        self._unavailable_reason = ""

        try:
            import pytesseract

            self._tesseract = pytesseract
        except ImportError:
            self._unavailable_reason = (
                "pytesseract is not installed. "
                "Install it with: pip install pytesseract"
            )
            return

        # Check if tesseract is available
        try:
            self._tesseract.get_tesseract_version()
            self.available = True
        except Exception as e:
            self._unavailable_reason = (
                f"Tesseract OCR is not available: {e}. "
                "Install it with:\n"
                "  Ubuntu/Debian: sudo apt install tesseract-ocr tesseract-ocr-chi-sim\n"
                "  macOS: brew install tesseract tesseract-lang\n"
                "  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
            )

    def is_available(self) -> bool:
        """Check if OCR functionality is available."""
        return self.available

    def get_unavailable_reason(self) -> str:
        """Get the reason why OCR is not available."""
        return self._unavailable_reason

    def render_page_to_image(self, page: pymupdf.Page, dpi: int = 200) -> bytes:
        """
        Render a PDF page as an image (fallback when OCR is not available).

        Args:
            page: PyMuPDF page object
            dpi: Resolution for rendering

        Returns:
            Image data as bytes (PNG format)
        """
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")

    def extract_text_from_page(self, page: pymupdf.Page) -> List["TextBlock"]:
        """
        Extract text from a PDF page using OCR.

        Args:
            page: PyMuPDF page object

        Returns:
            List of TextBlock objects with OCR text and positions
            Returns empty list if OCR is not available.
        """
        # Check if OCR is available
        if not self.available or self._tesseract is None:
            return []

        # Import here to avoid circular dependency
        from .pdf_parser import TextBlock, FontInfo

        # Render page to image
        pix = page.get_pixmap(dpi=200)
        img_data = pix.tobytes("png")

        # Open with PIL
        image = Image.open(io.BytesIO(img_data))

        # Get OCR data with bounding boxes
        try:
            ocr_data = self._tesseract.image_to_data(
                image, lang=self.lang, output_type=self._tesseract.Output.DICT
            )
        except Exception:
            # Fallback to basic text extraction
            text = self._tesseract.image_to_string(image, lang=self.lang)
            # Return as single text block covering the whole page
            rect = page.rect
            return [
                TextBlock(
                    text=text.strip(),
                    bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                    font_info=FontInfo(),
                )
            ]

        # Group text by line and create text blocks
        text_blocks = []
        n_boxes = len(ocr_data["text"])

        i = 0
        while i < n_boxes:
            text = ocr_data["text"][i].strip()
            if text and int(ocr_data["conf"][i]) > 30:  # Confidence threshold
                # Get bounding box coordinates
                x = ocr_data["left"][i]
                y = ocr_data["top"][i]
                w = ocr_data["width"][i]
                h = ocr_data["height"][i]

                # Convert from image coordinates to PDF coordinates
                # Image is at 200 DPI, PDF is at 72 DPI
                scale = 72 / 200
                x0 = x * scale
                y0 = y * scale
                x1 = (x + w) * scale
                y1 = (y + h) * scale

                text_blocks.append(
                    TextBlock(
                        text=text,
                        bbox=(x0, y0, x1, y1),
                        font_info=FontInfo(size=10),  # Default font size for OCR text
                    )
                )
            i += 1

        return text_blocks

    def get_available_languages(self) -> List[str]:
        """
        Get list of available Tesseract languages.

        Returns:
            List of language codes
        """
        try:
            return self._tesseract.get_languages()
        except Exception:
            return ["eng", "chi_sim"]  # Default fallback

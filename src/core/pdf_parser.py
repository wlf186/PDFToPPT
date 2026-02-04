"""PDF element extraction module."""

import io
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pymupdf


@dataclass
class FontInfo:
    """Font information for text blocks."""

    font: str = ""
    size: float = 12.0
    color: Tuple[int, int, int] = (0, 0, 0)  # RGB
    bold: bool = False
    italic: bool = False


@dataclass
class TextBlock:
    """A text block extracted from a PDF page."""

    text: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    font_info: FontInfo = field(default_factory=FontInfo)


@dataclass
class ImageBlock:
    """An image block extracted from a PDF page."""

    image_bytes: bytes
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)


@dataclass
class TableBlock:
    """A table block detected in a PDF page."""

    rows: int
    cols: int
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    data: List[List[str]]


class PDFElementExtractor:
    """Extract elements (text, images, tables) from PDF pages."""

    def __init__(self, use_ocr: bool = False, ocr_lang: str = "chi_sim+eng"):
        """
        Initialize the extractor.

        Args:
            use_ocr: Whether to use OCR for text extraction
            ocr_lang: OCR language(s) to use
        """
        self.use_ocr = use_ocr
        self.ocr_lang = ocr_lang
        self._ocr_handler = None
        self._ocr_available = False

        if use_ocr:
            from .ocr_handler import OCRHandler

            self._ocr_handler = OCRHandler(ocr_lang)
            self._ocr_available = self._ocr_handler.is_available()

    def is_ocr_available(self) -> bool:
        """Check if OCR is available and functional."""
        return self._ocr_available

    def get_ocr_unavailable_reason(self) -> str:
        """Get the reason why OCR is not available."""
        if self._ocr_handler:
            return self._ocr_handler.get_unavailable_reason()
        return "OCR not requested"

    def render_page_as_image(self, page: pymupdf.Page, dpi: int = 200) -> bytes:
        """
        Render a PDF page as an image (for scanned PDFs without OCR).

        Args:
            page: PyMuPDF page object
            dpi: Resolution for rendering

        Returns:
            Image data as bytes (PNG format)
        """
        if self._ocr_handler:
            return self._ocr_handler.render_page_to_image(page, dpi)

        # Fallback: render directly
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")

    def extract_text_blocks(self, page: pymupdf.Page) -> List[TextBlock]:
        """
        Extract text blocks from a PDF page.

        Args:
            page: PyMuPDF page object

        Returns:
            List of TextBlock objects
        """
        text_blocks = []

        # Get text in dictionary format with coordinates
        text_dict = page.get_text("dict")

        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:  # Text block
                bbox = block.get("bbox", (0, 0, 0, 0))

                # Skip if bbox is invalid
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    continue

                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue

                        # Calculate precise bbox for this span
                        span_bbox = span.get("bbox", bbox)

                        # Extract font information
                        font_info = FontInfo(
                            font=span.get("font", ""),
                            size=span.get("size", 12.0),
                            color=self._rgb_to_tuple(span.get("color", 0)),
                            bold=span.get("flags", 0) & 2**4 != 0,
                            italic=span.get("flags", 0) & 2**1 != 0,
                        )

                        text_blocks.append(
                            TextBlock(text=text, bbox=span_bbox, font_info=font_info)
                        )

        # If no text found and OCR is enabled and available, use OCR
        if not text_blocks and self._ocr_available and self._ocr_handler:
            text_blocks = self._ocr_handler.extract_text_from_page(page)

        return text_blocks

    def extract_images(self, page: pymupdf.Page) -> List[ImageBlock]:
        """
        Extract images from a PDF page.

        Args:
            page: PyMuPDF page object

        Returns:
            List of ImageBlock objects
        """
        image_blocks = []
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]

            # Get image bounding box
            try:
                # Get the image area on the page
                img_rects = page.get_image_rects(xref)
                if not img_rects:
                    continue

                # Combine all rects for this image
                bbox = self._combine_rects(img_rects)

                # Extract image bytes
                base_image = page.parent.extract_image(xref)
                if base_image:
                    image_bytes = base_image["image"]
                    image_blocks.append(ImageBlock(image_bytes=image_bytes, bbox=bbox))

            except Exception:
                # Skip images that can't be extracted
                continue

        return image_blocks

    def detect_tables(
        self, page: pymupdf.Page, text_blocks: List[TextBlock]
    ) -> List[TableBlock]:
        """
        Detect tables in a PDF page based on text alignment.

        Args:
            page: PyMuPDF page object
            text_blocks: List of text blocks on the page

        Returns:
            List of TableBlock objects
        """
        tables = []

        # Try to find tables using PyMuPDF's table detection
        try:
            tabs = page.find_tables()
            for tab in tabs:
                if tab.header_present:
                    bbox = tab.bbox
                    df = tab.to_pandas()

                    # Convert DataFrame to list of lists
                    data = df.values.tolist()
                    # Convert headers to list
                    data.insert(0, df.columns.tolist())

                    tables.append(
                        TableBlock(
                            rows=len(data),
                            cols=len(data[0]) if data else 0,
                            bbox=bbox,
                            data=data,
                        )
                    )
        except Exception:
            # If table detection fails, return empty list
            pass

        return tables

    @staticmethod
    def _rgb_to_tuple(color_int: int) -> Tuple[int, int, int]:
        """
        Convert integer color value to RGB tuple.

        Args:
            color_int: Color as integer (0xBBGGRR format)

        Returns:
            RGB tuple
        """
        r = color_int & 0xFF
        g = (color_int >> 8) & 0xFF
        b = (color_int >> 16) & 0xFF
        return (r, g, b)

    @staticmethod
    def _combine_rects(rects: list) -> Tuple[float, float, float, float]:
        """
        Combine multiple rectangles into a single bounding box.

        Args:
            rects: List of pymupdf.Rect objects

        Returns:
            Combined bounding box as (x0, y0, x1, y1)
        """
        if not rects:
            return (0, 0, 0, 0)

        x0 = min(rect.x0 for rect in rects)
        y0 = min(rect.y0 for rect in rects)
        x1 = max(rect.x1 for rect in rects)
        y1 = max(rect.y1 for rect in rects)

        return (x0, y0, x1, y1)

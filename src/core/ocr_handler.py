"""OCR text extraction module using multimodal LLM."""

import base64
import io
import logging
from typing import List, Optional

import pymupdf
from PIL import Image

from .config import get_config
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


class OCRHandler:
    """Handle OCR text extraction using multimodal LLM."""

    def __init__(self, lang: str = "auto"):
        """
        Initialize the OCR handler.

        Args:
            lang: Language hint for OCR (default: auto-detect)
                   Not used for LLM but kept for API compatibility
        """
        self.lang = lang
        self._llm_client: Optional[LLMClient] = None
        self.available = False
        self._unavailable_reason = ""

        # Try to initialize LLM client
        config = get_config()
        if config.llm.enabled and config.llm.is_configured():
            self._llm_client = LLMClient(config.llm)
            if self._llm_client.is_available():
                self.available = True
            else:
                self._unavailable_reason = self._llm_client.get_error()
        else:
            self._unavailable_reason = (
                "LLM is not configured. Please configure LLM in config.yaml "
                "to use OCR features."
            )

    def is_available(self) -> bool:
        """Check if OCR functionality is available."""
        return self.available

    def get_unavailable_reason(self) -> str:
        """Get the reason why OCR is not available."""
        return self._unavailable_reason

    def render_page_to_image(self, page: pymupdf.Page, dpi: int = 200) -> bytes:
        """
        Render a PDF page as an image.

        Args:
            page: PyMuPDF page object
            dpi: Resolution for rendering

        Returns:
            Image data as bytes (PNG format)
        """
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")

    def _encode_image_to_base64(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64 data URL format."""
        base64_str = base64.b64encode(image_bytes).decode("utf-8")

        # Detect image type from magic bytes
        if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            mime_type = "image/png"
        elif image_bytes[:2] == b'\xff\xd8':
            mime_type = "image/jpeg"
        else:
            mime_type = "image/png"

        return f"data:{mime_type};base64,{base64_str}"

    def extract_text_from_page(self, page: pymupdf.Page) -> List["TextBlock"]:
        """
        Extract text from a PDF page using multimodal LLM OCR.

        Args:
            page: PyMuPDF page object

        Returns:
            List of TextBlock objects with OCR text and positions
            Returns empty list if OCR is not available.
        """
        # Import here to avoid circular dependency
        from .pdf_parser import TextBlock, FontInfo

        # Check if LLM is available
        if not self.available or self._llm_client is None:
            logger.warning("OCR requested but LLM is not available")
            return []

        # Render page to image
        image_bytes = self.render_page_to_image(page)
        image_url = self._encode_image_to_base64(image_bytes)

        # Build OCR prompt
        prompt = self._build_ocr_prompt()

        # Call LLM for OCR
        try:
            response = self._llm_client._client.chat.completions.create(
                model=self._llm_client.config.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                max_tokens=4096,
                temperature=0.1,
                timeout=self._llm_client.config.timeout,
            )

            content = response.choices[0].message.content

            # Parse the LLM response into text blocks
            return self._parse_ocr_response(content, page)

        except Exception as e:
            logger.error(f"LLM OCR failed: {e}")
            # Fallback: return empty list (caller will use full page image)
            return []

    def _build_ocr_prompt(self) -> str:
        """Build prompt for LLM OCR."""
        return """Extract all text from this document page with precise position information.

Please analyze the image and provide a JSON response with the following structure:
{
    "text_blocks": [
        {
            "text": "the extracted text content",
            "bbox": {"x0": 0, "y0": 0, "x1": 100, "y1": 20},
            "type": "heading|paragraph|list_item|table_cell"
        }
    ]
}

Requirements:
1. Extract ALL visible text in the image
2. For each text block, provide:
   - text: the exact text content (preserve line breaks within blocks)
   - bbox: bounding box in pixels [x0, y0, x1, y1] from top-left
   - type: the type of text content
3. Group related text lines into logical blocks
4. Preserve reading order (top to bottom, left to right)
5. Include headers, paragraphs, lists, table contents

Respond ONLY with valid JSON, no additional text."""

    def _parse_ocr_response(self, content: str, page: pymupdf.Page) -> List["TextBlock"]:
        """Parse LLM OCR response into TextBlock objects."""
        from .pdf_parser import TextBlock, FontInfo
        import json

        try:
            # Try to parse as JSON
            if content.strip().startswith("{"):
                data = json.loads(content)

                if "text_blocks" in data:
                    text_blocks = []
                    page_rect = page.rect

                    # Get image dimensions for coordinate conversion
                    pix = page.get_pixmap(dpi=200)
                    img_width = pix.width
                    img_height = pix.height
                    pdf_width = page_rect.width
                    pdf_height = page_rect.height

                    scale_x = pdf_width / img_width
                    scale_y = pdf_height / img_height

                    for block in data["text_blocks"]:
                        bbox = block.get("bbox", {})
                        x0 = bbox.get("x0", 0) * scale_x
                        y0 = bbox.get("y0", 0) * scale_y
                        x1 = bbox.get("x1", img_width) * scale_x
                        y1 = bbox.get("y1", img_height) * scale_y

                        text_blocks.append(
                            TextBlock(
                                text=block["text"],
                                bbox=(x0, y0, x1, y1),
                                font_info=FontInfo(size=11),
                            )
                        )

                    return text_blocks

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug(f"Failed to parse LLM OCR JSON response: {e}")

        # Fallback: treat entire response as a single text block
        # Clean up common JSON/markdown artifacts
        text = content.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:-1])
        if text.startswith("{") or text.startswith("["):
            try:
                data = json.loads(text)
                if isinstance(data, dict) and "text_blocks" in data:
                    return self._parse_ocr_response(json.dumps(data), page)
            except:
                pass

        # Last resort: create a single text block covering the page
        rect = page.rect
        return [
            TextBlock(
                text=text,
                bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                font_info=FontInfo(size=11),
            )
        ]

    def extract_text_simple(self, page: pymupdf.Page) -> str:
        """
        Extract text from page as a simple string (no position info).

        Args:
            page: PyMuPDF page object

        Returns:
            Extracted text as string
        """
        # Try PyMuPDF's built-in text extraction first
        text = page.get_text()
        if text.strip():
            return text

        # If no text found and LLM is available, use LLM OCR
        if self.available and self._llm_client:
            image_bytes = self.render_page_to_image(page)
            image_url = self._encode_image_to_base64(image_bytes)

            try:
                response = self._llm_client._client.chat.completions.create(
                    model=self._llm_client.config.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Extract all text from this image. Preserve paragraphs and line breaks. Respond only with the extracted text, no additional commentary.",
                                },
                                {"type": "image_url", "image_url": {"url": image_url}},
                            ],
                        }
                    ],
                    max_tokens=4096,
                    temperature=0.1,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"LLM text extraction failed: {e}")

        return ""

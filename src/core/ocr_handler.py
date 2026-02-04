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

    def _get_ocr_dpi(self) -> int:
        """Get OCR DPI from config or use sensible default."""
        config = get_config()
        # Check if using ollama preset (slow local model)
        if config.llm.preset == "ollama":
            return 100  # Lower DPI for faster processing
        return 150  # Higher DPI for faster cloud models

    def render_page_to_image(self, page: pymupdf.Page, dpi: Optional[int] = None) -> bytes:
        """
        Render a PDF page as an image.

        Args:
            page: PyMuPDF page object
            dpi: Resolution for rendering (if None, uses auto-detected value)

        Returns:
            Image data as bytes (PNG format)
        """
        if dpi is None:
            dpi = self._get_ocr_dpi()
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

        # Check if using ollama - use simpler mode for local models
        config = get_config()
        use_simple_mode = config.llm.preset == "ollama"

        # Render page to image with appropriate DPI
        dpi = self._get_ocr_dpi()
        image_bytes = self.render_page_to_image(page, dpi=dpi)
        image_url = self._encode_image_to_base64(image_bytes)

        # Log image size for debugging
        logger.debug(f"OCR image size: {len(image_bytes) / 1024:.1f} KB at {dpi} DPI")

        # Build OCR prompt (simpler for ollama)
        if use_simple_mode:
            prompt = self._build_simple_ocr_prompt()
        else:
            prompt = self._build_detailed_ocr_prompt()

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
                max_tokens=2048 if use_simple_mode else 4096,
                temperature=0.1,
                timeout=self._llm_client.config.timeout,
            )

            content = response.choices[0].message.content

            # Parse the LLM response into text blocks
            if use_simple_mode:
                return self._parse_simple_ocr_response(content, page)
            else:
                return self._parse_detailed_ocr_response(content, page)

        except Exception as e:
            logger.error(f"LLM OCR failed: {e}")
            # Fallback: return empty list (caller will use full page image)
            return []

    def _build_simple_ocr_prompt(self) -> str:
        """Build simplified prompt for slower local models."""
        return """Extract all text from this image line by line.

Format your response as a simple list:
Line 1: [text]
Line 2: [text]
...

Just extract the visible text in reading order. No positioning data needed."""

    def _build_detailed_ocr_prompt(self) -> str:
        """Build detailed prompt with position information."""
        return """Extract all text from this document page with precise position information.

Please analyze the image and provide a JSON response with the following structure:
{
    "text_blocks": [
        {
            "text": "the extracted text content",
            "bbox": {"x0": 0, "y0": 0, "x1": 100, "y1": 20}
        }
    ]
}

Requirements:
1. Extract ALL visible text in the image
2. Group related text lines into logical blocks
3. Preserve reading order (top to bottom, left to right)

Respond ONLY with valid JSON, no additional text."""

    def _parse_simple_ocr_response(self, content: str, page: pymupdf.Page) -> List["TextBlock"]:
        """Parse simple OCR response (line by line) into TextBlock objects."""
        from .pdf_parser import TextBlock, FontInfo

        # Clean up response
        text = content.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:-1])

        # Remove common prefixes
        lines = []
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("Line") and ":" in line:
                line = line.split(":", 1)[1].strip()
            if line:
                lines.append(line)

        if not lines:
            # Fallback: treat entire response as one block
            rect = page.rect
            return [TextBlock(text=text, bbox=(rect.x0, rect.y0, rect.x1, rect.y1), font_info=FontInfo(size=11))]

        # Create text blocks distributed vertically on the page
        rect = page.rect
        page_height = rect.y1 - rect.y0
        block_height = min(page_height / len(lines), 30)  # Max 30 points per line

        text_blocks = []
        for i, line in enumerate(lines):
            y0 = rect.y0 + (i * block_height)
            y1 = min(y0 + block_height, rect.y1)
            text_blocks.append(
                TextBlock(
                    text=line,
                    bbox=(rect.x0, y0, rect.x1, y1),
                    font_info=FontInfo(size=11),
                )
            )

        return text_blocks

    def _parse_detailed_ocr_response(self, content: str, page: pymupdf.Page) -> List["TextBlock"]:
        """Parse detailed LLM OCR response into TextBlock objects."""
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
                    pix = page.get_pixmap(dpi=self._get_ocr_dpi())
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

        # Fallback to simple parsing
        return self._parse_simple_ocr_response(content, page)

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
                    max_tokens=2048,
                    temperature=0.1,
                    timeout=self._llm_client.config.timeout,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"LLM text extraction failed: {e}")

        return ""

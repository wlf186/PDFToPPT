"""LLM client for PDF to PPT conversion enhancement."""

import base64
import logging
from typing import Any, Dict, List, Optional

from .config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for interacting with LLM API services using OpenAI-compatible interface."""

    def __init__(self, config: LLMConfig):
        """
        Initialize LLM client.

        Args:
            config: LLM configuration
        """
        self.config = config
        self._available = False
        self._error_message: Optional[str] = None
        self._client = None

        # Check if configured
        if not config.is_configured():
            self._error_message = "LLM not configured (set base_url and model_name in config.yaml)"
            return

        # Try to import and initialize OpenAI client
        try:
            from openai import OpenAI

            # Initialize OpenAI client with custom base_url
            self._client = OpenAI(
                base_url=config.base_url,
                api_key=config.api_key or "dummy-key",  # Some services require a key
            )
            self._available = True
        except ImportError:
            self._error_message = "openai not installed. Install with: pip install openai"
        except Exception as e:
            self._error_message = f"Failed to initialize OpenAI client: {str(e)}"

    def is_available(self) -> bool:
        """Check if LLM client is available."""
        return self._available

    def get_error(self) -> Optional[str]:
        """Get error message if not available."""
        return self._error_message

    def get_info(self) -> str:
        """Get LLM info for display."""
        if self._available:
            return f"{self.config.model_name} @ {self.config.base_url}"
        return f"Not available: {self._error_message}"

    def _encode_image_to_base64(self, image_bytes: bytes) -> str:
        """
        Encode image bytes to base64 data URL format.

        Args:
            image_bytes: Image data as bytes

        Returns:
            Base64 encoded data URL string
        """
        base64_str = base64.b64encode(image_bytes).decode("utf-8")

        # Detect image type from magic bytes
        if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            mime_type = "image/png"
        elif image_bytes[:2] == b'\xff\xd8':
            mime_type = "image/jpeg"
        else:
            # Default to PNG for PyMuPDF rendered images
            mime_type = "image/png"

        return f"data:{mime_type};base64,{base64_str}"

    def enhance_page(
        self,
        page_image: bytes,
        page_number: int,
        extracted_text: str,
    ) -> Dict[str, Any]:
        """
        Use LLM to analyze and enhance page content.

        Args:
            page_image: Page image as bytes (PNG/JPEG)
            page_number: Page number (1-indexed)
            extracted_text: Extracted text from the page

        Returns:
            Dictionary with enhanced layout and content information
        """
        if not self._available or self._client is None:
            return {"error": self._error_message}

        try:
            # Encode image to base64 data URL
            image_url = self._encode_image_to_base64(page_image)

            # Prepare prompt
            prompt = self._build_prompt(page_number, extracted_text)

            # Call LLM API using OpenAI client
            response = self._client.chat.completions.create(
                model=self.config.model_name,
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
                timeout=self.config.timeout,
            )

            # Parse response
            content = response.choices[0].message.content

            # Try to parse JSON response
            import json

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # If not JSON, return as text
                return {"raw_response": content}

        except Exception as e:
            logger.warning(f"LLM call failed for page {page_number}: {e}")
            return {"error": str(e)}

    def _build_prompt(self, page_number: int, extracted_text: str) -> str:
        """Build prompt for LLM."""
        return f"""You are a PDF to PowerPoint conversion assistant. Analyze this page image and extract structured content for creating a PPT slide.

Page {page_number} - Already extracted text:
{extracted_text if extracted_text else "(No text extracted)"}

Please analyze the image and provide a JSON response with the following structure:
{{
    "title": "Main title or heading of this slide",
    "content_blocks": [
        {{
            "type": "heading|paragraph|list|table|image",
            "text": "Content text",
            "position": {{"x0": 0, "y0": 0, "x1": 100, "y1": 20}},
            "style": {{"font_size": 18, "bold": true, "color": [0,0,0]}}
        }}
    ],
    "layout": "title_top|two_column|blank|other",
    "background_color": [255, 255, 255]
}}

Focus on:
1. Correct text hierarchy and grouping
2. Proper positioning of elements
3. Font styles and sizes
4. Color information
5. Table structures if present

Respond ONLY with valid JSON, no additional text."""

    def test_connection(self) -> tuple[bool, str]:
        """
        Test LLM connection.

        Returns:
            Tuple of (success, message)
        """
        if not self._available or self._client is None:
            return False, self._error_message or "Not configured"

        try:
            # Simple test request
            response = self._client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=10,
                timeout=10,
            )

            return True, f"Connected to {self.config.model_name} @ {self.config.base_url}"

        except Exception as e:
            return False, f"Connection failed: {str(e)}"

    def enhance_page_sync(
        self,
        page_image: bytes,
        page_number: int,
        extracted_text: str,
    ) -> Dict[str, Any]:
        """
        Synchronous version of enhance_page (already sync with OpenAI client).

        Args:
            page_image: Page image as bytes
            page_number: Page number
            extracted_text: Extracted text

        Returns:
            Dictionary with enhanced layout information
        """
        return self.enhance_page(page_image, page_number, extracted_text)

    def test_connection_sync(self) -> tuple[bool, str]:
        """Synchronous version of test_connection (already sync with OpenAI client)."""
        return self.test_connection()

"""LLM client for PDF to PPT conversion enhancement."""

import base64
import io
import logging
from typing import Any, Dict, List, Optional

from .config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for interacting with LLM API services."""

    def __init__(self, config: LLMConfig):
        """
        Initialize LLM client.

        Args:
            config: LLM configuration
        """
        self.config = config
        self._available = False
        self._error_message: Optional[str] = None

        # Check if configured
        if not config.is_configured():
            self._error_message = "LLM not configured (set base_url and model_name in config.yaml)"
            return

        # Try to import httpx
        try:
            import httpx

            self._httpx = httpx
            self._available = True
        except ImportError:
            self._error_message = "httpx not installed. Install with: pip install httpx"

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

    async def enhance_page(
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
        if not self._available:
            return {"error": self._error_message}

        try:
            # Encode image to base64
            image_base64 = base64.b64encode(page_image).decode("utf-8")
            image_url = f"data:image/png;base64,{image_base64}"

            # Prepare prompt
            prompt = self._build_prompt(page_number, extracted_text)

            # Call LLM API
            response = await self._call_llm(
                images=[image_url],
                prompt=prompt,
            )

            return response

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

    async def _call_llm(
        self,
        images: List[str],
        prompt: str,
    ) -> Dict[str, Any]:
        """
        Call LLM API.

        Args:
            images: List of image URLs (base64 data URLs)
            prompt: Text prompt

        Returns:
            Parsed response as dictionary
        """
        import httpx

        # Build messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *[{"type": "image_url", "image_url": {"url": img}} for img in images],
                ],
            }
        ]

        # Make API request
        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        timeout = httpx.Timeout(self.config.timeout, connect=10.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{self.config.base_url.rstrip('/')}/chat/completions",
                json={
                    "model": self.config.model_name,
                    "messages": messages,
                    "max_tokens": 4096,
                    "temperature": 0.1,
                },
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

        # Parse response
        content = data["choices"][0]["message"]["content"]

        # Try to parse JSON response
        import json

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # If not JSON, return as text
            return {"raw_response": content}

    async def test_connection(self) -> tuple[bool, str]:
        """
        Test LLM connection.

        Returns:
            Tuple of (success, message)
        """
        if not self._available:
            return False, self._error_message or "Not configured"

        try:
            import httpx

            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            timeout = httpx.Timeout(10.0, connect=5.0)

            async with httpx.AsyncClient(timeout=timeout) as client:
                # Simple test request
                response = await client.post(
                    f"{self.config.base_url.rstrip('/')}/chat/completions",
                    json={
                        "model": self.config.model_name,
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 10,
                    },
                    headers=headers,
                )

            if response.status_code == 200:
                return True, f"Connected to {self.config.model_name} @ {self.config.base_url}"
            else:
                return False, f"API returned status {response.status_code}: {response.text[:200]}"

        except Exception as e:
            return False, f"Connection failed: {str(e)}"

    def enhance_page_sync(
        self,
        page_image: bytes,
        page_number: int,
        extracted_text: str,
    ) -> Dict[str, Any]:
        """
        Synchronous version of enhance_page.

        Args:
            page_image: Page image as bytes
            page_number: Page number
            extracted_text: Extracted text

        Returns:
            Dictionary with enhanced layout information
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.enhance_page(page_image, page_number, extracted_text)
        )

    def test_connection_sync(self) -> tuple[bool, str]:
        """Synchronous version of test_connection."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.test_connection())

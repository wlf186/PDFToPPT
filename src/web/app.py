"""FastAPI web application for PDF to PPT conversion."""

import io
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from ..core import PDFToPPTConverter

app = FastAPI(title="PDF to PPT Converter", version="1.0.0")

# Get the directory of this file
BASE_DIR = Path(__file__).parent


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main web interface."""
    template_path = BASE_DIR / "templates" / "index.html"
    if not template_path.exists():
        return HTMLResponse(content="<h1>Template not found</h1>", status_code=404)
    return HTMLResponse(content=template_path.read_text(encoding="utf-8"))


@app.post("/convert")
async def convert_pdf(
    file: UploadFile = File(..., description="PDF file to convert"),
    ocr: bool = Form(False, description="Use OCR for scanned PDFs"),
    ocr_lang: str = Form("chi_sim+eng", description="OCR language(s)"),
):
    """
    Convert an uploaded PDF file to PPTX.

    Returns the converted PPTX file as a download.
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Save uploaded PDF
        pdf_path = temp_dir_path / file.filename
        try:
            content = await file.read()
            pdf_path.write_bytes(content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save PDF: {e}")

        # Generate output filename
        output_filename = pdf_path.stem + ".pptx"
        ppt_path = temp_dir_path / output_filename

        # Perform conversion
        try:
            converter = PDFToPPTConverter(
                pdf_path=pdf_path,
                ppt_path=ppt_path,
                use_ocr=ocr,
                ocr_lang=ocr_lang,
            )
            converter.convert()
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Conversion failed: {str(e)}"
            )

        # Return the converted file
        if not ppt_path.exists():
            raise HTTPException(status_code=500, detail="Conversion failed - no output file")

        return FileResponse(
            path=str(ppt_path),
            filename=output_filename,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        )


@app.post("/info")
async def get_pdf_info(file: UploadFile = File(...)):
    """
    Get information about an uploaded PDF file.

    Returns page count, dimensions, and metadata.
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        pdf_path = temp_dir_path / file.filename

        try:
            content = await file.read()
            pdf_path.write_bytes(content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save PDF: {e}")

        try:
            info_data = PDFToPPTConverter.get_pdf_info(pdf_path)
            return {
                "filename": file.filename,
                "page_count": info_data["page_count"],
                "width": info_data["width"],
                "height": info_data["height"],
                "metadata": info_data.get("metadata", {}),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read PDF: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "pdf-to-ppt-converter"}


def run_server(host: str = "127.0.0.1", port: int = 8000):
    """
    Run the FastAPI server.

    Args:
        host: Host to bind to
        port: Port to bind to
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

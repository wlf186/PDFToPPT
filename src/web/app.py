"""FastAPI web application for PDF to PPT conversion."""

import io
import json
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Optional
from queue import Queue

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from core import PDFToPPTConverter

logger = logging.getLogger(__name__)

app = FastAPI(title="PDF to PPT Converter", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the directory of this file
BASE_DIR = Path(__file__).parent

# Store progress queues for SSE
progress_queues: dict[str, Queue] = {}


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main web interface."""
    template_path = BASE_DIR / "templates" / "index.html"
    if not template_path.exists():
        return HTMLResponse(content="<h1>Template not found</h1>", status_code=404)
    return HTMLResponse(content=template_path.read_text(encoding="utf-8"))


@app.get("/favicon.ico")
async def favicon():
    """Return empty response for favicon to avoid 404 errors."""
    from fastapi.responses import Response
    # Return a tiny 1x1 transparent PNG as favicon
    favicon_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82"
    return Response(content=favicon_data, media_type="image/x-icon")


@app.get("/progress/{job_id}")
async def progress_stream(job_id: str):
    """Server-Sent Events endpoint for real-time progress updates."""
    from fastapi.responses import StreamingResponse

    if job_id not in progress_queues:
        progress_queues[job_id] = Queue()

    queue = progress_queues[job_id]

    async def event_generator():
        try:
            while True:
                # Check for new progress updates
                if not queue.empty():
                    data = queue.get()
                    if data is None:  # End of stream signal
                        break
                    yield f"data: {json.dumps(data)}\n\n"
                else:
                    # Send heartbeat every second
                    yield ": heartbeat\n\n"
                    import asyncio
                    await asyncio.sleep(0.5)
        finally:
            # Clean up queue after connection closes
            progress_queues.pop(job_id, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/llm-status")
async def get_llm_status():
    """
    Get LLM configuration and connection status.

    Returns information about whether LLM enhancement is available.
    """
    status = PDFToPPTConverter.get_global_llm_status()
    return status


@app.post("/convert")
async def convert_pdf(
    file: UploadFile = File(..., description="PDF file to convert"),
    ocr: bool = Form(False, description="Use OCR for scanned PDFs"),
    ocr_lang: str = Form("chi_sim+eng", description="OCR language(s)"),
    use_llm: bool = Form(True, description="Use LLM enhancement if available"),
):
    """
    Convert an uploaded PDF file to PPTX.

    Returns the converted PPTX file as a download.
    """
    # Generate unique job ID for progress tracking
    job_id = str(uuid.uuid4())

    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Check LLM status for logging
    if use_llm:
        llm_status = PDFToPPTConverter.get_global_llm_status()
        if llm_status.get("available"):
            logger.info(f"LLM enhancement enabled: {llm_status.get('model')} @ {llm_status.get('base_url')}")
        elif llm_status.get("enabled"):
            logger.warning(f"LLM enabled but not available: {llm_status.get('message')}")
        else:
            logger.info("LLM enhancement is disabled")

    # Create progress queue for this job
    progress_queues[job_id] = Queue()

    # Setup progress callback that sends to queue
    def progress_callback(current: int, total: int, using_llm: bool, message: str):
        """Send progress updates to queue."""
        progress = int((current / total) * 100) if total > 0 else 0
        progress_data = {
            "current": current,
            "total": total,
            "progress": progress,
            "using_llm": using_llm,
            "message": message,
        }
        # Non-blocking put
        if job_id in progress_queues:
            try:
                progress_queues[job_id].put_nowait(progress_data)
            except:
                pass

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
            logger.info(f"[{file.filename}] Starting conversion...")
            converter = PDFToPPTConverter(
                pdf_path=pdf_path,
                ppt_path=ppt_path,
                use_ocr=ocr,
                ocr_lang=ocr_lang,
                use_llm=use_llm,
                progress_callback=progress_callback,
            )
            converter.convert()
            logger.info(f"[{file.filename}] Conversion completed successfully")
        except Exception as e:
            logger.error(f"[{file.filename}] Conversion failed: {e}")
            # Send error via progress queue
            if job_id in progress_queues:
                progress_queues[job_id].put_nowait({"error": str(e)})
            raise HTTPException(
                status_code=500, detail=f"Conversion failed: {str(e)}"
            )

        # Read the converted file into memory before temp dir is deleted
        if not ppt_path.exists():
            raise HTTPException(status_code=500, detail="Conversion failed - no output file")

        ppt_bytes = ppt_path.read_bytes()

    # Signal completion and clean up queue
    if job_id in progress_queues:
        progress_queues[job_id].put_nowait({"complete": True})
        progress_queues[job_id].put_nowait(None)  # End signal
        del progress_queues[job_id]

    # Return the file from memory (after temp dir is cleaned up)
    return StreamingResponse(
        io.BytesIO(ppt_bytes),
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={
            "Content-Disposition": f"attachment; filename={output_filename}",
            "X-Job-ID": job_id,  # Return job ID for client to track progress
        },
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
    """Health check endpoint with LLM status."""
    llm_status = PDFToPPTConverter.get_global_llm_status()

    return {
        "status": "healthy",
        "service": "pdf-to-ppt-converter",
        "llm": {
            "enabled": llm_status.get("enabled", False),
            "available": llm_status.get("available", False),
            "model": llm_status.get("model"),
            "base_url": llm_status.get("base_url"),
        },
    }


def run_server(host: str = "127.0.0.1", port: int = 8000):
    """
    Run the FastAPI server.

    Args:
        host: Host to bind to
        port: Port to bind to
    """
    import uvicorn

    # Configure logging to show progress
    logging_config = uvicorn.config.LOGGING_CONFIG
    logging_config["formatters"]["default"]["fmt"] = "%(levelname)s: %(message)s"

    uvicorn.run(app, host=host, port=port, log_config=logging_config)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

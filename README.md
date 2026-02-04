# PDF to PPT Converter

<div align="center">

A Python tool that converts PDF files to editable PowerPoint presentations using multimodal LLM for OCR and layout enhancement.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Features](#features) | [Installation](#installation) | [LLM Setup](#llm-setup) | [Usage](#usage)

</div>

---

## Features

- **Text Extraction** - Extracts text blocks with exact positioning
- **Image Support** - Preserves images from PDF
- **Table Detection** - Automatically detects and converts tables
- **LLM-Powered OCR** - Uses multimodal LLM for scanned PDF text extraction
- **Layout Enhancement** - Optional LLM analysis for improved slide layout
- **Dual Interface** - Both CLI and Web interface
- **Progress Tracking** - Real-time progress bar with LLM status

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure LLM service (see LLM Setup section below)
# Edit config.yaml and set enabled: true

# Convert PDF to PPT
python run.py convert input.pdf -o output.pptx

# Or start web interface
python run.py web
```

---

## Installation

```bash
# Clone from GitHub
git clone https://github.com/wlf186/PDFToPPT.git
cd PDFToPPT

# Install dependencies
pip install -r requirements.txt
```

---

## LLM Setup

This tool requires a multimodal LLM service for OCR and layout enhancement. You can use:

### Option 1: Ollama (Recommended, Free)

```bash
# Install Ollama from https://ollama.ai
# Pull a multimodal model
ollama pull qwen3-vl:4b
# or
ollama pull qwen2.5-vl:7b
```

Edit `config.yaml`:
```yaml
llm:
  preset: ollama
  model_name: qwen3-vl:4b  # or qwen2.5-vl:7b, llava:7b
  enabled: true
```

### Option 2: OpenAI-Compatible API

Edit `config.yaml`:
```yaml
llm:
  preset: custom
  base_url: "http://your-server:8000/v1"
  api_key: "your-api-key"
  model_name: "qwen3-vl-32b-instruct"
  enabled: true
  no_proxy: "your-server-domain"  # For internal services
```

### Available Presets

| Preset | Base URL | Default Model |
|--------|----------|---------------|
| `ollama` | `http://localhost:11434/v1` | `qwen3-vl:4b` |
| `openai` | `https://api.openai.com/v1` | `gpt-4o` |
| `custom` | (manual) | (manual) |

### Check LLM Status

```bash
python run.py llm-status
```

Output:
```
=== LLM Status ===

Enabled: Yes
Configured: Yes
Available: Yes
Model: qwen3-vl:4b [ollama]
Base URL: http://localhost:11434/v1
```

---

## Usage

### CLI Mode

**Basic conversion:**
```bash
python run.py convert input.pdf -o output.pptx
```

**With progress bar:**
```bash
python run.py convert input.pdf -o output.pptx
# Shows: Converting: 50%|████▌     | 5/10 [00:15<00:15, 3.1s/page, Page 5/10 done [LLM]]
```

**Disable LLM enhancement:**
```bash
python run.py convert input.pdf --no-llm
```

**Get PDF information:**
```bash
python run.py info input.pdf
```

### Web Mode

```bash
python run.py web
# Opens at http://localhost:8000
```

Web interface shows:
- LLM connection status
- Real-time progress in console logs
- File upload and download

---

## Configuration

Edit `config.yaml` to customize:

```yaml
llm:
  # Quick preset selection (ollama, openai, custom)
  preset: "ollama"

  # Override preset values if needed
  model_name: "qwen2.5-vl:7b"

  # Enable LLM features
  enabled: true

  # For internal services, bypass proxy
  no_proxy: "localhost"

  # Timeout for API calls
  timeout: 60
```

---

## How It Works

1. **PDF Parsing** - Uses PyMuPDF to extract elements
2. **OCR for Scanned PDFs** - Multimodal LLM extracts text from images
3. **Layout Enhancement** - Optional LLM analysis for better slide structure
4. **Coordinate Conversion** - Converts PDF points to PPTX EMU
5. **PPT Generation** - Creates PPTX file with python-pptx

```
PDF → [PyMuPDF] → Elements → [LLM OCR] → Text + [LLM Enhancement] → Layout → [python-pptx] → PPTX
```

---

## Project Structure

```
pdf-to-ppt/
├── src/
│   ├── core/
│   │   ├── converter.py       # Core conversion logic
│   │   ├── pdf_parser.py      # PDF element extraction
│   │   ├── ppt_builder.py     # PPT construction
│   │   ├── ocr_handler.py     # LLM-based OCR
│   │   ├── llm_client.py      # LLM API client
│   │   └── config.py          # Configuration management
│   ├── web/
│   │   ├── app.py             # FastAPI application
│   │   └── templates/
│   │       └── index.html     # Web interface
│   └── cli.py                 # Command-line interface
├── config.yaml                # LLM configuration
├── requirements.txt
└── run.py                     # Entry point
```

---

## Requirements

- Python 3.8+
- Multimodal LLM service (Ollama, OpenAI, or compatible)
- Dependencies:
  - `pymupdf` - PDF parsing
  - `python-pptx` - PPT generation
  - `openai` - LLM API client
  - `pydantic` - Configuration
  - `pyyaml` - Config file parsing

---

## FAQ

### Q: Is this tool completely free?
**A:** Yes, if you use Ollama with local models. It runs entirely on your machine.

### Q: Can I use this offline?
**A:** Yes, with Ollama running locally.

### Q: What if I don't configure an LLM?
**A:** The tool will still work for PDFs with embedded text. Scanned PDFs will render as images.

### Q: Which LLM model should I use?
**A:**
- **Local (free)**: `qwen3-vl:4b`, `qwen2.5-vl:7b`, `llava:7b`
- **Cloud**: `gpt-4o`, `claude-3.5-sonnet` (via compatible API)

### Q: How long does conversion take?
**A:** Depends on PDF size and LLM speed. With local Ollama: ~2-5 seconds per page.

### Q: Can I convert password-protected PDFs?
**A:** No, remove password protection first.

---

## License

MIT License - feel free to use this tool for any purpose.

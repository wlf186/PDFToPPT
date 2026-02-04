# PDF to PPT Converter

<div align="center">

A completely local Python tool that converts PDF files to editable PowerPoint presentations while preserving element positions.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Features](#features) | [Installation](#installation) | [Usage](#usage) | [FAQ](#faq)

</div>

---

## Features

- **Text Extraction** - Extracts text blocks with exact positioning
- **Image Support** - Preserves images from PDF
- **Table Detection** - Automatically detects and converts tables
- **OCR Support** - Optional Tesseract OCR for scanned PDFs
- **Dual Interface** - Both CLI and Web interface
- **Local Processing** - No external services required, 100% offline
- **Smart Fallback** - Scanned PDFs render as images when OCR is unavailable

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Convert PDF to PPT
python run.py convert input.pdf -o output.pptx

# Or start web interface
python run.py web
```

---

## Installation

### Method 1: Clone from GitHub

```bash
git clone https://github.com/wlf186/PDFToPPT.git
cd PDFToPPT
pip install -r requirements.txt
```

### Method 2: Download ZIP

1. Download the repository as ZIP from GitHub
2. Extract to a folder
3. Open terminal/command prompt in that folder
4. Run: `pip install -r requirements.txt`

---

## Platform-Specific Setup

### Windows

#### Prerequisites
1. Install Python 3.8+ from [python.org](https://www.python.org/downloads/)
   - During installation, check **"Add Python to PATH"**

2. Install dependencies:
   ```cmd
   pip install -r requirements.txt
   ```

#### Optional: OCR Support (for scanned PDFs)
1. Download Tesseract installer:
   - [64-bit Windows](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install to default location: `C:\Program Files\Tesseract-OCR`
3. Restart your command prompt

#### Run on Windows
```cmd
# CLI mode
python run.py convert input.pdf -o output.pptx

# Web mode
python run.py web
```

---

### macOS

#### Prerequisites
```bash
# Install Python 3 (if not installed)
brew install python3

# Install dependencies
pip3 install -r requirements.txt
```

#### Optional: OCR Support
```bash
brew install tesseract tesseract-lang
```

---

### Linux (Ubuntu/Debian)

```bash
# Install Python and pip
sudo apt update
sudo apt install python3 python3-pip

# Install dependencies
pip3 install -r requirements.txt

# Optional: OCR Support
sudo apt install tesseract-ocr tesseract-ocr-chi-sim
```

---

## Usage

### CLI Mode

**Basic conversion:**
```bash
python run.py convert input.pdf -o output.pptx
```

**Convert with OCR (for scanned PDFs):**
```bash
python run.py convert input.pdf -o output.pptx --ocr
```

**Specify OCR language:**
```bash
python run.py convert input.pdf --ocr --ocr-lang chi_sim
```

**Get PDF information:**
```bash
python run.py info input.pdf
```

---

### Web Mode

**Start the web server:**
```bash
python run.py web
```

Then open your browser to `http://localhost:8000`

**Custom host/port:**
```bash
python run.py web --host 0.0.0.0 --port 8080
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
│   │   └── ocr_handler.py     # OCR text recognition
│   ├── web/
│   │   ├── app.py             # FastAPI application
│   │   └── templates/
│   │       └── index.html     # Web interface
│   └── cli.py                 # Command-line interface
├── requirements.txt
├── README.md
└── run.py                     # Entry point
```

---

## How It Works

1. **PDF Parsing** - Uses PyMuPDF to extract text, images, and table structures
2. **Coordinate Conversion** - Converts PDF coordinates (points) to PPTX coordinates (EMU)
3. **Element Mapping**:
   - Text blocks → Text boxes with font preservation
   - Images → Picture shapes
   - Tables → Table structures
4. **PPT Generation** - Creates PPTX file with python-pptx

### Smart Fallback for Scanned PDFs

If you don't have Tesseract OCR installed, scanned PDFs will be automatically rendered as images instead of failing. This ensures the tool works even without OCR dependencies.

---

## OCR Languages

Supported languages depend on Tesseract installation. Common ones:

| Language | Code |
|----------|------|
| Chinese (Simplified) | `chi_sim` |
| Chinese (Traditional) | `chi_tra` |
| English | `eng` |
| Japanese | `jpn` |
| Korean | `kor` |

Combine multiple languages with `+`: `chi_sim+eng`

---

## FAQ

### Q: Do I need to install Tesseract OCR?
**A:** Only if you want to extract text from scanned PDFs. For regular PDFs with embedded text, it's not required.

### Q: What happens if I try to convert a scanned PDF without OCR?
**A:** The page will be rendered as an image in the PPT. You can view it, but the text won't be editable.

### Q: Does this work offline?
**A:** Yes, completely. No internet connection required.

### Q: Can I convert password-protected PDFs?
**A:** No, you need to remove the password protection first.

### Q: The conversion is slow for large files. Is this normal?
**A:** Yes, processing time depends on file size and complexity. Files with many images may take longer.

### Q: Windows says "python is not recognized"
**A:** Make sure you checked "Add Python to PATH" during installation, or use `py` instead of `python`.

---

## Limitations

- PDF password protection must be removed before conversion
- Complex layouts may not be perfectly preserved
- Fonts may be substituted if not available in PPT
- Large files (>50MB) may take longer to process
- Vector graphics are converted to images

---

## Requirements

- Python 3.8 or higher
- Dependencies listed in `requirements.txt`:
  - pymupdf (PDF parsing)
  - python-pptx (PPT generation)
  - fastapi + uvicorn (Web server)
  - click (CLI)
  - pytesseract (OCR, optional)

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## License

MIT License - feel free to use this tool for any purpose.

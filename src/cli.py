"""Command-line interface for PDF to PPT converter."""

import sys
from pathlib import Path

import click

from .core import PDFToPPTConverter


@click.group()
def cli():
    """PDF to PPT converter - Convert PDF files to editable PowerPoint presentations."""
    pass


@cli.command()
@click.argument("pdf_file", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output PPTX file path (default: same as input with .pptx extension)",
)
@click.option("--ocr", is_flag=True, help="Use OCR for scanned PDFs")
@click.option(
    "--ocr-lang",
    default="chi_sim+eng",
    help="OCR language (default: chi_sim+eng for Chinese and English)",
)
def convert(pdf_file, output, ocr, ocr_lang):
    """
    Convert a PDF file to PPTX.

    Examples:

        # Basic conversion
        python run.py convert input.pdf -o output.pptx

        # Convert with OCR for scanned PDFs
        python run.py convert input.pdf -o output.pptx --ocr

        # Convert with specific OCR language
        python run.py convert input.pdf --ocr --ocr-lang chi_sim
    """
    pdf_path = Path(pdf_file)

    # Determine output path
    if output:
        ppt_path = Path(output)
    else:
        ppt_path = pdf_path.with_suffix(".pptx")

    click.echo(f"Converting {pdf_file} to {ppt_path}...")

    try:
        # Perform conversion
        converter = PDFToPPTConverter(
            pdf_path=pdf_path,
            ppt_path=ppt_path,
            use_ocr=ocr,
            ocr_lang=ocr_lang,
        )

        # Show progress
        page_count = converter.get_page_count()
        click.echo(f"Processing {page_count} page(s)...")

        result_path = converter.convert()

        click.echo(f"Successfully created: {result_path}", err=True)

    except Exception as e:
        click.echo(f"Error during conversion: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("pdf_file", type=click.Path(exists=True))
def info(pdf_file):
    """
    Show information about a PDF file.

    Examples:
        python run.py info input.pdf
    """
    pdf_path = Path(pdf_file)

    try:
        info_data = PDFToPPTConverter.get_pdf_info(pdf_path)

        click.echo(f"PDF Information for: {pdf_file}")
        click.echo(f"  Pages: {info_data['page_count']}")
        click.echo(f"  Dimensions: {info_data['width']:.2f} x {info_data['height']:.2f} points")
        click.echo(f"  Dimensions: {info_data['width']/72:.2f} x {info_data['height']/72:.2f} inches")

        metadata = info_data.get("metadata", {})
        if metadata.get("title"):
            click.echo(f"  Title: {metadata['title']}")
        if metadata.get("author"):
            click.echo(f"  Author: {metadata['author']}")

    except Exception as e:
        click.echo(f"Error reading PDF: {e}", err=True)
        sys.exit(1)


def main():
    """Main entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()

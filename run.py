"""Main entry point for PDF to PPT converter."""

import sys
from pathlib import Path

import click

# Add src directory to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))


@click.group()
def cli():
    """PDF to PPT Converter - Convert PDF files to editable PowerPoint presentations."""
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
@click.option("--no-llm", is_flag=True, help="Disable LLM enhancement")
@click.option("--no-progress", is_flag=True, help="Disable progress bar")
def convert(pdf_file, output, ocr, ocr_lang, no_llm, no_progress):
    """
    Convert a PDF file to PPTX.

    Examples:

        # Basic conversion
        python run.py convert input.pdf -o output.pptx

        # Convert with OCR for scanned PDFs
        python run.py convert input.pdf -o output.pptx --ocr

        # Convert with specific OCR language
        python run.py convert input.pdf --ocr --ocr-lang chi_sim

        # Disable LLM enhancement
        python run.py convert input.pdf --no-llm

        # Disable progress bar
        python run.py convert input.pdf --no-progress
    """
    from core import PDFToPPTConverter

    pdf_path = Path(pdf_file)

    # Determine output path
    if output:
        ppt_path = Path(output)
    else:
        ppt_path = pdf_path.with_suffix(".pptx")

    click.echo(f"Converting {pdf_file} to {ppt_path}...")

    # Check LLM status
    use_llm = not no_llm
    if use_llm:
        llm_status = PDFToPPTConverter.get_global_llm_status()
        if llm_status.get("available"):
            click.echo(click.style(f"✓ {llm_status['message']}", fg="green"))
        elif llm_status.get("enabled"):
            click.echo(click.style(f"⚠ LLM: {llm_status['message']}", fg="yellow"))
        else:
            click.echo(click.style(f"ℹ {llm_status['message']}", fg="blue"))

    try:
        # Setup progress callback
        progress_bar = None
        last_message = [""]

        def progress_callback(current: int, total: int, using_llm: bool, message: str):
            nonlocal progress_bar, last_message

            if no_progress:
                if message and message != last_message[0]:
                    if using_llm:
                        click.echo(f"  {message} " + click.style("[LLM]", fg="cyan"))
                    else:
                        click.echo(f"  {message}")
                    last_message[0] = message
                return

            if progress_bar is None:
                # Initialize progress bar
                from tqdm import tqdm

                progress_bar = tqdm(
                    total=total,
                    desc="Converting",
                    unit="page",
                    colour="green",
                )

            # Update progress bar
            progress_bar.n = current
            llm_tag = " [LLM]" if using_llm else ""
            progress_bar.set_postfix_str(f"{message}{llm_tag}")
            progress_bar.refresh()

        # Perform conversion
        converter = PDFToPPTConverter(
            pdf_path=pdf_path,
            ppt_path=ppt_path,
            use_ocr=ocr,
            ocr_lang=ocr_lang,
            use_llm=use_llm,
            progress_callback=progress_callback,
        )

        # Get page count
        page_count = converter.get_page_count()
        if no_progress:
            click.echo(f"Processing {page_count} page(s)...")

        result_path = converter.convert()

        # Close progress bar
        if progress_bar is not None:
            progress_bar.close()

        # Show output location
        click.echo("")
        click.echo(click.style(f"✓ Successfully created:", fg="green", bold=True))
        click.echo(f"  {result_path}")

    except Exception as e:
        if progress_bar is not None:
            progress_bar.close()
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
    from core import PDFToPPTConverter

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


@cli.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
def web(host, port):
    """
    Start the web server.

    Examples:
        python run.py web
        python run.py web --host 0.0.0.0 --port 8080
    """
    from core import PDFToPPTConverter
    from web.app import run_server

    # Check and display LLM status
    llm_status = PDFToPPTConverter.get_global_llm_status()

    click.echo(f"Starting web server at http://{host}:{port}")

    if llm_status.get("available"):
        click.echo(click.style(f"✓ LLM: {llm_status['model']} @ {llm_status['base_url']}", fg="green"))
    elif llm_status.get("enabled"):
        click.echo(click.style(f"⚠ LLM: {llm_status['message']}", fg="yellow"))
    else:
        click.echo(click.style(f"ℹ LLM enhancement is disabled", fg="blue"))

    run_server(host=host, port=port)


@cli.command()
def llm_status():
    """
    Check LLM configuration and connection status.

    Examples:
        python run.py llm-status
    """
    from core import PDFToPPTConverter

    status = PDFToPPTConverter.get_global_llm_status()

    click.echo("\n=== LLM Status ===\n")

    click.echo(f"Enabled: {click.style('Yes' if status.get('enabled') else 'No', fg='green' if status.get('enabled') else 'blue')}")
    click.echo(f"Configured: {click.style('Yes' if status.get('configured') else 'No', fg='green' if status.get('configured') else 'yellow')}")

    if status.get("available"):
        click.echo(f"Available: {click.style('Yes', fg='green')}")
        click.echo(f"Model: {status.get('model', 'N/A')}")
        click.echo(f"Base URL: {status.get('base_url', 'N/A')}")
    else:
        click.echo(f"Available: {click.style('No', fg='red')}")
        if status.get("error"):
            click.echo(f"Error: {status['error']}")

    click.echo(f"\nMessage: {status.get('message', 'N/A')}")
    click.echo("")


if __name__ == "__main__":
    cli()

from pathlib import Path
import json
import logging

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

# -------------------------------------------------------
# Logging
# -------------------------------------------------------

logging.basicConfig(level=logging.ERROR)

# -------------------------------------------------------
# Project paths
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_PAPERS = PROJECT_ROOT / "data" / "raw_papers"
PARSED_PAPERS = PROJECT_ROOT / "data" / "parsed_papers"

PARSED_PAPERS.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------
# Configure Docling
# -------------------------------------------------------

pipeline_options = PdfPipelineOptions()
pipeline_options.generate_picture_images = True

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    }
)

# -------------------------------------------------------
# Find PDFs
# -------------------------------------------------------

pdf_files = sorted(RAW_PAPERS.glob("*.pdf"))

print(f"\nFound {len(pdf_files)} PDF(s).\n")

# -------------------------------------------------------
# Parse PDFs
# -------------------------------------------------------

for index, pdf_path in enumerate(pdf_files, start=1):

    print("=" * 70)
    print(f"[{index}/{len(pdf_files)}] {pdf_path.name}")
    print("=" * 70)

    try:

        paper_name = pdf_path.stem

        output_dir = PARSED_PAPERS / paper_name
        output_dir.mkdir(parents=True, exist_ok=True)

        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)

        # --------------------------------------------
        # Convert
        # --------------------------------------------

        result = converter.convert(str(pdf_path))
        doc = result.document

        # --------------------------------------------
        # Markdown
        # --------------------------------------------

        markdown_path = output_dir / f"{paper_name}.md"

        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(doc.export_to_markdown())

        # --------------------------------------------
        # JSON
        # --------------------------------------------

        json_path = output_dir / f"{paper_name}.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                doc.export_to_dict(),
                f,
                indent=2,
                ensure_ascii=False,
            )

        # --------------------------------------------
        # Images
        # --------------------------------------------

        image_count = 0
        saved_images = 0

        for item, _ in doc.iterate_items():

            if getattr(item, "label", None) != "picture":
                continue

            image_count += 1

            if hasattr(item, "get_image"):

                image = item.get_image(doc)

                if image is not None:

                    image_path = images_dir / f"figure_{image_count}.png"

                    image.save(image_path)

                    saved_images += 1

        print(f"Markdown : {markdown_path.name}")
        print(f"JSON      : {json_path.name}")
        print(f"Images    : {saved_images}/{image_count}")

    except Exception as e:

        print(f"Failed to parse {pdf_path.name}")
        print(e)

print("\nDone.")
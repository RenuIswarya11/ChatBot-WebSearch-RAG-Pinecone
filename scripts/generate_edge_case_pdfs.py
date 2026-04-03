"""Generate oversized PDFs to test upload page-limit validation."""

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas


def draw_page_lines(pdf: canvas.Canvas, lines: list[str], x: int, y: int, width: int) -> None:
    """Draw wrapped lines on a single PDF page.

    Args:
        pdf (canvas.Canvas): ReportLab canvas used for writing.
        lines (list[str]): Text lines to place on the page.
        x (int): Left x-coordinate for rendering text.
        y (int): Starting y-coordinate for rendering text.
        width (int): Maximum width in points before wrapping.

    Returns:
        None: Text is drawn directly on the provided canvas.
    """
    current_y = y
    for line in lines:
        wrapped = simpleSplit(line, "Helvetica", 11, width)
        for segment in wrapped:
            pdf.drawString(x, current_y, segment)
            current_y -= 16
        current_y -= 2


def main() -> None:
    """Generate two 12-page PDFs for edge-case validation.

    Args:
        None: Uses fixed file names and writes output files.

    Returns:
        None: Files are written to `test_documents`.
    """
    docs = [
        ("edge_case_policy_manual_12_pages.pdf", "Policy Manual - Edge Case"),
        ("edge_case_research_appendix_12_pages.pdf", "Research Appendix - Edge Case"),
    ]

    output_dir = Path("test_documents")
    output_dir.mkdir(exist_ok=True)

    for file_name, title in docs:
        pdf = canvas.Canvas(str(output_dir / file_name), pagesize=LETTER)
        for page in range(1, 13):
            pdf.setFont("Helvetica-Bold", 14)
            pdf.drawString(72, 740, title)
            pdf.setFont("Helvetica", 11)
            chapter = (page - 1) // 3 + 1
            pdf.drawString(72, 718, f"Chapter {chapter} - Page {page} of 12")
            lines = [
                "This edge-case document intentionally exceeds the 10-page upload limit.",
                "It is designed to validate strict page-count guardrails in the upload flow.",
                f"Current file: {file_name}. Current page: {page}.",
                "Policy note: oversized uploads should be rejected before indexing begins.",
                "Expected app behavior: show clear error and avoid vector upsert.",
                "Testing outcome should be deterministic across repeated uploads.",
            ]
            if "policy_manual" in file_name:
                lines.extend(
                    [
                        "Governance section: define ownership for model and data incidents.",
                        "Compliance section: record audit logs for all user-facing outputs.",
                    ]
                )
            else:
                lines.extend(
                    [
                        "Appendix section: include ablation references and prompt variants.",
                        "Appendix section: preserve experiment metadata for reproducibility.",
                    ]
                )
            draw_page_lines(pdf, lines, x=72, y=690, width=460)
            pdf.showPage()
        pdf.save()

    print("Generated 2 edge-case PDFs (12 pages each).")


if __name__ == "__main__":
    main()


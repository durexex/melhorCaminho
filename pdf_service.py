import logging
from fpdf import FPDF

logger = logging.getLogger(__name__)

_PAGE_WIDTH_MM = 210
_MARGIN_MM = 15
_CONTENT_WIDTH_MM = _PAGE_WIDTH_MM - 2 * _MARGIN_MM

_FONT_FAMILY = "Helvetica"
_TITLE_SIZE = 18
_HEADING_SIZE = 14
_SUBHEADING_SIZE = 12
_BODY_SIZE = 11
_LINE_HEIGHT = 6


def _parse_heading_level(line: str) -> tuple[int, str]:
    """Return (heading_level, stripped_text). Level 0 means regular text."""
    stripped = line.lstrip()
    level = 0
    while level < len(stripped) and stripped[level] == "#":
        level += 1
    if level == 0 or level >= len(stripped) or stripped[level] != " ":
        return 0, line
    return min(level, 3), stripped[level + 1 :].strip()


def _strip_bold_markers(text: str) -> str:
    return text.replace("**", "")


def _sanitize_text(text: str) -> str:
    """Replace characters outside latin-1 with '?' to avoid encoding errors."""
    return text.encode("latin-1", "replace").decode("latin-1")


def create_pdf_report(title: str, content: str, filename: str) -> bytes:
    """Generate a formatted PDF report and return the raw bytes.

    Args:
        title: Report title displayed as the PDF header.
        content: Body text, may contain basic Markdown (headings, bold).
        filename: Logical filename (metadata only; bytes are returned).

    Returns:
        Raw PDF bytes ready for download or file write.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_left_margin(_MARGIN_MM)
    pdf.set_right_margin(_MARGIN_MM)
    pdf.set_title(title)
    pdf.set_creator("MelhorCaminho - Assistente IA")
    pdf.add_page()
    safe_title = _sanitize_text(_strip_bold_markers(title))
    pdf.set_font(_FONT_FAMILY, "B", _TITLE_SIZE)
    pdf.cell(0, 12, safe_title, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)
    y = pdf.get_y()
    pdf.line(_MARGIN_MM, y, _PAGE_WIDTH_MM - _MARGIN_MM, y)
    pdf.ln(6)
    _render_content(pdf, content)
    return bytes(pdf.output())


def _render_content(pdf: FPDF, content: str) -> None:
    """Parse content line-by-line and write formatted text into the PDF."""
    for line in content.split("\n"):
        level, text = _parse_heading_level(line)
        text = _sanitize_text(_strip_bold_markers(text))
        if level == 1:
            pdf.set_font(_FONT_FAMILY, "B", _HEADING_SIZE)
            pdf.ln(4)
            pdf.multi_cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        elif level >= 2:
            pdf.set_font(_FONT_FAMILY, "B", _SUBHEADING_SIZE)
            pdf.ln(3)
            pdf.multi_cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        elif text.strip() == "":
            pdf.ln(_LINE_HEIGHT)
        else:
            pdf.set_font(_FONT_FAMILY, "", _BODY_SIZE)
            pdf.multi_cell(0, _LINE_HEIGHT, text, new_x="LMARGIN", new_y="NEXT")

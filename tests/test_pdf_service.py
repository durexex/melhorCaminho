import pytest

from pdf_service import (
    create_pdf_report,
    _parse_heading_level,
    _strip_bold_markers,
    _sanitize_text,
)

SAMPLE_CONTENT = (
    "# Relatório de Eficiência\n"
    "\n"
    "## Resumo da Rota\n"
    "Número de pontos: 5\n"
    "Distância total: 1234.56 km\n"
    "\n"
    "## Análise\n"
    "A rota otimizada apresenta uma **boa distribuição** entre os pontos.\n"
    "### Sugestões\n"
    "- Considerar agrupamento regional.\n"
)


class TestParseHeadingLevel:
    def test_h1(self):
        level, text = _parse_heading_level("# Título Principal")
        assert level == 1
        assert text == "Título Principal"

    def test_h2(self):
        level, text = _parse_heading_level("## Subtítulo")
        assert level == 2
        assert text == "Subtítulo"

    def test_h3(self):
        level, text = _parse_heading_level("### Seção")
        assert level == 3
        assert text == "Seção"

    def test_deep_heading_capped_at_3(self):
        level, _ = _parse_heading_level("#### Muito Profundo")
        assert level == 3

    def test_regular_line(self):
        level, text = _parse_heading_level("Texto comum")
        assert level == 0
        assert text == "Texto comum"

    def test_hash_without_space_is_regular(self):
        level, text = _parse_heading_level("#semespaço")
        assert level == 0
        assert text == "#semespaço"

    def test_empty_line(self):
        level, text = _parse_heading_level("")
        assert level == 0
        assert text == ""


class TestStripBoldMarkers:
    def test_removes_double_asterisks(self):
        assert _strip_bold_markers("**negrito**") == "negrito"

    def test_no_markers(self):
        assert _strip_bold_markers("texto normal") == "texto normal"

    def test_mixed(self):
        result = _strip_bold_markers("Isso é **importante** e **urgente**")
        assert result == "Isso é importante e urgente"


class TestSanitizeText:
    def test_portuguese_accents_preserved(self):
        text = "Ação rápida: café, não, ônibus, útil"
        assert _sanitize_text(text) == text

    def test_replaces_unsupported_chars(self):
        result = _sanitize_text("Emoji: \U0001f600")
        assert "\U0001f600" not in result


class TestCreatePdfReport:
    def test_returns_bytes(self):
        result = create_pdf_report("Título", "Conteúdo", "relatorio.pdf")
        assert isinstance(result, bytes)

    def test_not_empty(self):
        result = create_pdf_report("Título", "Conteúdo", "relatorio.pdf")
        assert len(result) > 0

    def test_starts_with_pdf_header(self):
        result = create_pdf_report("Test", "Body", "test.pdf")
        assert result[:5] == b"%PDF-"

    def test_title_in_metadata(self):
        result = create_pdf_report("Simple Title", "corpo", "r.pdf")
        assert b"Simple Title" in result

    def test_generates_larger_pdf_with_more_content(self):
        short = create_pdf_report("T", "x", "a.pdf")
        long_content = "Linha de conteudo. " * 50
        long = create_pdf_report("T", long_content, "b.pdf")
        assert len(long) > len(short)

    def test_accented_characters(self):
        result = create_pdf_report(
            "Instruções de Navegação",
            "Próxima parada: São Paulo\nAção recomendada: café",
            "nav.pdf",
        )
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_multiline_markdown_content(self):
        result = create_pdf_report("Report", SAMPLE_CONTENT, "report.pdf")
        assert isinstance(result, bytes)
        assert len(result) > 100

    def test_empty_content(self):
        result = create_pdf_report("Empty", "", "empty.pdf")
        assert result[:5] == b"%PDF-"

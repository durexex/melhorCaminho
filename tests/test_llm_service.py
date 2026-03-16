import pytest
from unittest.mock import patch, MagicMock

from llm_service import LLMService, _format_route_data


SAMPLE_ROUTE_DATA = {
    "cities": [(100, 200), (300, 400), (500, 600)],
    "sequence": [0, 2, 1],
    "total_distance": 565.68,
    "num_cities": 3,
    "priorities": {"0": "critical_meds", "1": "regular_supplies", "2": "critical_meds"},
}


@pytest.fixture
def route_data():
    return dict(SAMPLE_ROUTE_DATA)


class TestFormatRouteData:
    def test_returns_all_keys(self, route_data):
        result = _format_route_data(route_data)
        assert "num_cities" in result
        assert "total_distance" in result
        assert "sequence" in result
        assert "cities" in result
        assert "priorities" in result

    def test_num_cities_fallback(self):
        data = {"cities": [(0, 0), (1, 1)], "total_distance": 1.41}
        result = _format_route_data(data)
        assert result["num_cities"] == 2

    def test_defaults_for_missing_keys(self):
        result = _format_route_data({})
        assert result["num_cities"] == 0
        assert result["total_distance"] == 0.0
        assert result["sequence"] == []
        assert result["cities"] == []
        assert result["priorities"] == "N/A"


@patch("llm_service.ChatOpenAI")
class TestLLMServiceInit:
    def test_creates_llm_with_correct_params(self, mock_chat):
        mock_chat.return_value = MagicMock()
        svc = LLMService(api_key="test-key", model="gpt-4o")
        mock_chat.assert_called_once_with(
            api_key="test-key",
            model="gpt-4o",
            temperature=0.3,
        )

    def test_default_model(self, mock_chat):
        mock_chat.return_value = MagicMock()
        LLMService(api_key="test-key")
        mock_chat.assert_called_once_with(
            api_key="test-key",
            model="gpt-4",
            temperature=0.3,
        )


@patch("llm_service.ChatOpenAI")
class TestNavigationInstructions:
    def test_invokes_chain_and_returns_string(self, mock_chat, route_data):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Instruções geradas")
        mock_chat.return_value = mock_llm

        svc = LLMService(api_key="k")
        svc._nav_chain = MagicMock()
        svc._nav_chain.invoke.return_value = "Passo 1: Siga para o Ponto 0 (100, 200)"

        result = svc.generate_navigation_instructions(route_data)
        assert isinstance(result, str)
        assert len(result) > 0
        svc._nav_chain.invoke.assert_called_once()

    def test_prompt_receives_formatted_data(self, mock_chat, route_data):
        mock_chat.return_value = MagicMock()
        svc = LLMService(api_key="k")
        svc._nav_chain = MagicMock()
        svc._nav_chain.invoke.return_value = "ok"

        svc.generate_navigation_instructions(route_data)

        call_args = svc._nav_chain.invoke.call_args[0][0]
        assert call_args["num_cities"] == 3
        assert call_args["total_distance"] == 565.68
        assert call_args["sequence"] == [0, 2, 1]

    def test_raises_runtime_error_on_failure(self, mock_chat, route_data):
        mock_chat.return_value = MagicMock()
        svc = LLMService(api_key="k")
        svc._nav_chain = MagicMock()
        svc._nav_chain.invoke.side_effect = Exception("API timeout")

        with pytest.raises(RuntimeError, match="Falha ao gerar instruções de navegação"):
            svc.generate_navigation_instructions(route_data)


@patch("llm_service.ChatOpenAI")
class TestEfficiencyReport:
    def test_invokes_chain_and_returns_string(self, mock_chat, route_data):
        mock_chat.return_value = MagicMock()
        svc = LLMService(api_key="k")
        svc._efficiency_chain = MagicMock()
        svc._efficiency_chain.invoke.return_value = "## Relatório de Eficiência\n..."

        result = svc.generate_efficiency_report(route_data)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_prompt_receives_formatted_data(self, mock_chat, route_data):
        mock_chat.return_value = MagicMock()
        svc = LLMService(api_key="k")
        svc._efficiency_chain = MagicMock()
        svc._efficiency_chain.invoke.return_value = "ok"

        svc.generate_efficiency_report(route_data)

        call_args = svc._efficiency_chain.invoke.call_args[0][0]
        assert call_args["num_cities"] == 3
        assert call_args["total_distance"] == 565.68

    def test_raises_runtime_error_on_failure(self, mock_chat, route_data):
        mock_chat.return_value = MagicMock()
        svc = LLMService(api_key="k")
        svc._efficiency_chain = MagicMock()
        svc._efficiency_chain.invoke.side_effect = Exception("Rate limit")

        with pytest.raises(RuntimeError, match="Falha ao gerar relatório de eficiência"):
            svc.generate_efficiency_report(route_data)


@patch("llm_service.ChatOpenAI")
class TestChatResponse:
    def test_invokes_chat_chain_with_message(self, mock_chat, route_data):
        mock_chat.return_value = MagicMock()
        svc = LLMService(api_key="k")
        svc._chat_chain = MagicMock()
        svc._chat_chain.invoke.return_value = "A cidade mais distante é o Ponto 2."

        result = svc.chat_response("Qual a cidade mais distante?", route_data)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_passes_session_id_in_config(self, mock_chat, route_data):
        mock_chat.return_value = MagicMock()
        svc = LLMService(api_key="k")
        svc._chat_chain = MagicMock()
        svc._chat_chain.invoke.return_value = "ok"

        svc.chat_response("Olá", route_data, session_id="sess-42")

        _, kwargs = svc._chat_chain.invoke.call_args
        assert kwargs["config"]["configurable"]["session_id"] == "sess-42"

    def test_input_included_in_params(self, mock_chat, route_data):
        mock_chat.return_value = MagicMock()
        svc = LLMService(api_key="k")
        svc._chat_chain = MagicMock()
        svc._chat_chain.invoke.return_value = "ok"

        svc.chat_response("Quantas paradas?", route_data)

        call_args = svc._chat_chain.invoke.call_args[0][0]
        assert call_args["input"] == "Quantas paradas?"

    def test_raises_runtime_error_on_failure(self, mock_chat, route_data):
        mock_chat.return_value = MagicMock()
        svc = LLMService(api_key="k")
        svc._chat_chain = MagicMock()
        svc._chat_chain.invoke.side_effect = Exception("Connection refused")

        with pytest.raises(RuntimeError, match="Falha ao processar mensagem de chat"):
            svc.chat_response("Olá", route_data)


@patch("llm_service.ChatOpenAI")
class TestClearChatHistory:
    def test_clears_existing_session(self, mock_chat):
        mock_chat.return_value = MagicMock()
        svc = LLMService(api_key="k")
        svc._chat_history_store["sess-1"] = MagicMock()

        svc.clear_chat_history("sess-1")
        assert "sess-1" not in svc._chat_history_store

    def test_no_error_on_missing_session(self, mock_chat):
        mock_chat.return_value = MagicMock()
        svc = LLMService(api_key="k")
        svc.clear_chat_history("nonexistent")


@patch("llm_service.ChatOpenAI")
class TestSessionHistory:
    def test_creates_new_history_for_unknown_session(self, mock_chat):
        mock_chat.return_value = MagicMock()
        svc = LLMService(api_key="k")

        history = svc._get_session_history("new-session")
        assert history is not None
        assert "new-session" in svc._chat_history_store

    def test_returns_same_history_for_existing_session(self, mock_chat):
        mock_chat.return_value = MagicMock()
        svc = LLMService(api_key="k")

        h1 = svc._get_session_history("s1")
        h2 = svc._get_session_history("s1")
        assert h1 is h2

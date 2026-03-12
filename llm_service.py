import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

logger = logging.getLogger(__name__)

_NAVIGATION_SYSTEM_PROMPT = """\
Você é um assistente de logística especializado em instruções de navegação para motoristas.
Com base nos dados da rota otimizada fornecidos abaixo, gere um roteiro passo a passo claro \
e acionável para o motorista. Use linguagem direta e objetiva.

Regras:
- Refira-se aos pontos como "Ponto 0", "Ponto 1", etc., seguido das coordenadas (x, y).
- Indique a ordem de visita conforme a sequência fornecida.
- Quando disponível, mencione a prioridade de entrega de cada ponto.
- Não invente nomes de ruas ou locais que não constem nos dados.
- Ao final, inclua a distância total da rota.
"""

_NAVIGATION_USER_TEMPLATE = """\
Dados da rota:
- Número de cidades: {num_cities}
- Distância total: {total_distance:.2f}
- Sequência de visita (índices): {sequence}
- Coordenadas das cidades: {cities}
- Prioridades: {priorities}

Gere as instruções de navegação passo a passo."""

_EFFICIENCY_SYSTEM_PROMPT = """\
Você é um analista de logística especializado em eficiência de rotas.
Com base nos dados da rota otimizada fornecidos, gere um relatório analítico de eficiência.

O relatório deve incluir:
1. **Resumo da rota**: número de pontos, distância total.
2. **Métricas de eficiência**: distância média entre pontos, análise de distribuição.
3. **Análise qualitativa**: pontos fortes da rota, possíveis gargalos.
4. **Sugestões de melhoria**: recomendações para otimização futura.

Regras:
- Use apenas os dados fornecidos; não invente informações.
- Formate o relatório em Markdown com seções claras.
- Inclua valores numéricos quando disponíveis.
"""

_EFFICIENCY_USER_TEMPLATE = """\
Dados da rota:
- Número de cidades: {num_cities}
- Distância total: {total_distance:.2f}
- Sequência de visita (índices): {sequence}
- Coordenadas das cidades: {cities}
- Prioridades: {priorities}

Gere o relatório de eficiência."""

_CHAT_SYSTEM_PROMPT = """\
Você é um assistente inteligente de logística. Responda perguntas do usuário \
exclusivamente com base nos dados da rota fornecidos abaixo e em conhecimentos \
gerais de logística. Se a resposta não puder ser determinada a partir dos dados, \
informe isso educadamente.

Dados da rota atual:
- Número de cidades: {num_cities}
- Distância total: {total_distance:.2f}
- Sequência de visita (índices): {sequence}
- Coordenadas das cidades: {cities}
- Prioridades: {priorities}
"""


def _format_route_data(route_data: dict) -> dict:
    """Normaliza route_data para preenchimento dos templates."""
    return {
        "num_cities": route_data.get("num_cities", len(route_data.get("cities", []))),
        "total_distance": route_data.get("total_distance", 0.0),
        "sequence": route_data.get("sequence", []),
        "cities": route_data.get("cities", []),
        "priorities": route_data.get("priorities", "N/A"),
    }


class LLMService:
    """Encapsula a comunicação com a API da OpenAI via LangChain."""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self._llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=0.3,
        )
        self._parser = StrOutputParser()

        self._nav_chain = (
            ChatPromptTemplate.from_messages([
                ("system", _NAVIGATION_SYSTEM_PROMPT),
                ("human", _NAVIGATION_USER_TEMPLATE),
            ])
            | self._llm
            | self._parser
        )

        self._efficiency_chain = (
            ChatPromptTemplate.from_messages([
                ("system", _EFFICIENCY_SYSTEM_PROMPT),
                ("human", _EFFICIENCY_USER_TEMPLATE),
            ])
            | self._llm
            | self._parser
        )

        self._chat_history_store: dict[str, InMemoryChatMessageHistory] = {}

        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", _CHAT_SYSTEM_PROMPT),
            ("placeholder", "{history}"),
            ("human", "{input}"),
        ])
        base_chat_chain = chat_prompt | self._llm | self._parser

        self._chat_chain = RunnableWithMessageHistory(
            base_chat_chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def _get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self._chat_history_store:
            self._chat_history_store[session_id] = InMemoryChatMessageHistory()
        return self._chat_history_store[session_id]

    def generate_navigation_instructions(self, route_data: dict) -> str:
        """Gera instruções de navegação passo a passo a partir dos dados da rota."""
        try:
            params = _format_route_data(route_data)
            return self._nav_chain.invoke(params)
        except Exception as exc:
            logger.error("Erro ao gerar instruções de navegação: %s", exc)
            raise RuntimeError(
                f"Falha ao gerar instruções de navegação: {exc}"
            ) from exc

    def generate_efficiency_report(self, route_data: dict) -> str:
        """Gera relatório de eficiência da rota otimizada."""
        try:
            params = _format_route_data(route_data)
            return self._efficiency_chain.invoke(params)
        except Exception as exc:
            logger.error("Erro ao gerar relatório de eficiência: %s", exc)
            raise RuntimeError(
                f"Falha ao gerar relatório de eficiência: {exc}"
            ) from exc

    def chat_response(
        self,
        message: str,
        route_context: dict,
        session_id: str = "default",
    ) -> str:
        """Responde perguntas do usuário com memória conversacional."""
        try:
            params = _format_route_data(route_context)
            params["input"] = message
            config = {"configurable": {"session_id": session_id}}
            return self._chat_chain.invoke(params, config=config)
        except Exception as exc:
            logger.error("Erro no chat: %s", exc)
            raise RuntimeError(f"Falha ao processar mensagem de chat: {exc}") from exc

    def clear_chat_history(self, session_id: str = "default") -> None:
        """Limpa o histórico de chat de uma sessão específica."""
        self._chat_history_store.pop(session_id, None)

"""Geração do relatório técnico de insights via API da Anthropic."""
from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_MAX_TOKENS = 4096

SYSTEM_PROMPT = """Você é um analista de dados sênior responsável por redigir relatórios técnicos.
Você receberá o perfil estatístico agregado de um dataset em JSON — nunca os dados brutos.

Legenda dos campos: cv = coeficiente de variação (std/|mean|); skew = assimetria;
outliers_iqr = observações fora de [Q1 - 1,5·IQR, Q3 + 1,5·IQR]; pearson = correlação linear.

Regras:
- Baseie TODAS as afirmações exclusivamente no perfil fornecido e cite os valores que as sustentam.
- Não invente colunas, valores ou causas. Quando algo não puder ser concluído, diga isso explicitamente.
- Distinga correlação de causalidade e sinalize quando a qualidade dos dados limitar a conclusão.
- Escreva em português do Brasil, em tom técnico e objetivo, sem adjetivos promocionais.
- Responda em Markdown, usando exatamente estas seções:
## 1. Sumário executivo
## 2. Visão geral do dataset
## 3. Qualidade dos dados
## 4. Principais achados
## 5. Relações entre variáveis
## 6. Riscos e limitações
## 7. Recomendações (priorizadas e acionáveis)"""


class ReportError(Exception):
    """Falha na geração do relatório."""


def build_user_message(profile: dict[str, Any], context: str | None = None) -> str:
    payload = json.dumps(profile, ensure_ascii=False, separators=(",", ":"))
    sections = []
    if context:
        sections.append(f"Contexto de negócio informado pelo usuário:\n{context.strip()}")
    sections.append(f"Perfil estatístico do dataset (JSON):\n```json\n{payload}\n```")
    return "\n\n".join(sections)


class InsightReporter:
    """Transforma um perfil estatístico em um relatório técnico em Markdown."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = client or anthropic.Anthropic(max_retries=3, timeout=120.0)

    def generate(self, profile: dict[str, Any], context: str | None = None) -> str:
        message = build_user_message(profile, context)
        logger.info("Solicitando relatório ao modelo %s (%d caracteres).", self.model, len(message))
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": message}],
            )
        except anthropic.AuthenticationError as exc:
            raise ReportError("Chave de API inválida (verifique ANTHROPIC_API_KEY).") from exc
        except anthropic.APIError as exc:
            raise ReportError(f"Falha na chamada à API: {exc}") from exc

        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if not text:
            raise ReportError("A API retornou uma resposta sem conteúdo textual.")
        if response.stop_reason == "max_tokens":
            logger.warning("Relatório truncado pelo limite de %d tokens.", self.max_tokens)
        return text
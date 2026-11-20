# Agente de Análise Automática de Dados (LLM & Pandas)

## Sobre o Projeto
Ferramenta em Python desenvolvida para leitura automatizada de datasets operacionais, cálculo de estatísticas descritivas (matriz sintética) e integração com API de IA para geração de relatórios técnicos.

O projeto garante **privacidade de dados** (apenas agregações matematicamente sanitizadas são enviadas à API) e adota uma **arquitetura modular** orientada a objetos.

## Arquitetura e Tecnologias
- **Linguagem:** Python 3.12+
- **Processamento:** Pandas, NumPy.
- **Integração IA:** API da Anthropic (Claude).
- **Qualidade de Código:** Testes unitários automatizados com `pytest` e isolamento de segredos com `python-dotenv`.

## Como Executar
1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
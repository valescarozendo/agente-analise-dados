"""Interface de linha de comando do agente de análise de dados."""
from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from analista_dados import __version__
from analista_dados.loader import DatasetError, load_dataset
from analista_dados.profiler import build_profile
from analista_dados.reporter import DEFAULT_MODEL, InsightReporter, ReportError

logger = logging.getLogger("analista_dados")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analista_dados",
        description="Gera um relatório técnico de insights a partir de um dataset tabular.",
    )
    parser.add_argument(
        "dataset", type=Path, help="Arquivo de entrada (.csv, .tsv, .xlsx, .parquet, .json)."
    )
    parser.add_argument("-c", "--contexto", help="Contexto de negócio para orientar a análise.")
    parser.add_argument(
        "-o",
        "--saida",
        type=Path,
        help="Arquivo Markdown de saída (padrão: reports/relatorio_<dataset>_<data>.md).",
    )
    parser.add_argument(
        "-m",
        "--modelo",
        default=os.getenv("ANALISTA_MODEL", DEFAULT_MODEL),
        help="Modelo da API (padrão: %(default)s).",
    )
    parser.add_argument(
        "--salvar-perfil", type=Path, help="Salva em JSON a matriz estatística enviada à API."
    )
    parser.add_argument(
        "--somente-perfil",
        action="store_true",
        help="Calcula e exibe o perfil estatístico sem chamar a API.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Exibe logs detalhados.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(message)s",
    )
    try:
        return _run(args)
    except (DatasetError, ReportError) as exc:
        logger.error("%s", exc)
        return 1


def _run(args: argparse.Namespace) -> int:
    df = load_dataset(args.dataset)
    logger.info("Dataset carregado: %d linhas × %d colunas.", *df.shape)
    profile = build_profile(df)
    profile_json = json.dumps(profile, ensure_ascii=False, indent=2)

    if args.salvar_perfil:
        _write_text(args.salvar_perfil, profile_json)
        logger.info("Perfil salvo em %s", args.salvar_perfil)

    if args.somente_perfil:
        print(profile_json)
        return 0

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ReportError("ANTHROPIC_API_KEY não definida. Configure o arquivo .env.")

    report = InsightReporter(model=args.modelo).generate(profile, context=args.contexto)
    output = args.saida or _default_output_path(args.dataset)
    _write_text(output, _render_document(report, args.dataset, args.modelo))
    logger.info("Relatório salvo em %s", output)
    return 0


def _default_output_path(dataset: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("reports") / f"relatorio_{dataset.stem}_{stamp}.md"


def _render_document(report: str, dataset: Path, model: str) -> str:
    generated_at = datetime.now().strftime("%d/%m/%Y %H:%M")
    return (
        f"# Relatório de Análise — {dataset.name}\n\n"
        f"> Gerado em {generated_at} · Modelo: `{model}` · "
        f"Base: perfil estatístico agregado (sem dados brutos)\n\n"
        f"{report}\n"
    )


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
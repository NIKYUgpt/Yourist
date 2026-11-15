"""MCP stdio: только замороженные нормы, без сети, пользовательских данных и записи БД.

python -m app.evaluation.corpus_mcp --snapshot path/to/corpus-YYYY-MM-DD.json
Один процесс = один исследовательский сеанс с бюджетом 4 вызова.
"""

import argparse
from pathlib import Path

from app.evaluation.frozen_corpus import Snapshot
from app.modules.corpus.navigation import CorpusNavigator


def build_server(snapshot: Snapshot):
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations

    navigator = CorpusNavigator(snapshot)
    server = FastMCP("Yourist frozen corpus")
    annotations = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)

    @server.tool(annotations=annotations)
    def contents() -> dict:
        """Оглавление корпуса на зафиксированную дату."""
        return navigator.call("contents")

    @server.tool(annotations=annotations)
    def section(section_id: str) -> dict:
        """Статьи раздела по ID из contents."""
        return navigator.call("section", {"section_id": section_id})

    @server.tool(annotations=annotations)
    def article(act: str, key: str, offset: int = 0) -> dict:
        """Страница статьи; next_offset указывает продолжение, hash относится к целому тексту."""
        return navigator.call("article", {"act": act, "key": key, "offset": offset})

    @server.tool(annotations=annotations)
    def crossrefs(act: str, key: str) -> dict:
        """Кандидаты связей внутри акта; применимость и полнота не гарантируются."""
        return navigator.call("crossrefs", {"act": act, "key": key})

    @server.tool(annotations=annotations)
    def redaction(act: str) -> dict:
        """Редакция акта в этом снимке; это не полная история изменений."""
        return navigator.call("redaction", {"act": act})

    return server, navigator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    snapshot = Snapshot.model_validate_json(args.snapshot.read_bytes())
    server, _ = build_server(snapshot)
    server.run(transport="stdio")

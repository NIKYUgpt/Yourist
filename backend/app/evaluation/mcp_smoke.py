"""Проверка реального MCP stdio транспорта без вызовов моделей."""

import argparse
import asyncio
import importlib.metadata
import json
import sys
import time
from pathlib import Path

from app.evaluation.answer_dataset import write_json
from app.evaluation.frozen_corpus import Snapshot


async def run(snapshot_path: Path, out: Path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    if out.exists():
        raise ValueError("output exists")
    snap = Snapshot.model_validate_json(snapshot_path.read_bytes())
    snap.verify()
    article = next(a for a in snap.articles if not a.is_repealed and a.key == "70")
    report = {
        "snapshot_sha256": snap.sha256,
        "mcp_version": importlib.metadata.version("mcp"),
        "transport": "stdio",
        "calls": [],
    }
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.evaluation.corpus_mcp", "--snapshot", str(snapshot_path.resolve())],
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as client:
        initialized = await client.initialize()
        report["protocol"] = initialized.protocolVersion
        report["tools"] = [t.model_dump(mode="json") for t in (await client.list_tools()).tools]
        for tool, args in [
            ("contents", {}),
            ("article", {"act": article.act, "key": article.key}),
            ("article", {"act": article.act, "key": "999999"}),
            ("redaction", {"act": article.act}),
            ("contents", {}),
        ]:
            start = time.perf_counter()
            result = await client.call_tool(tool, args)
            report["calls"].append(
                {
                    "tool": tool,
                    "arguments": args,
                    "elapsed_ms": (time.perf_counter() - start) * 1000,
                    "result": result.model_dump(mode="json"),
                }
            )
    errors = [r["result"]["isError"] for r in report["calls"]]
    report["passed"] = errors == [False, False, True, False, True]
    write_json(out, report)
    if not report["passed"]:
        raise RuntimeError("unexpected MCP results; see artifact")
    print(json.dumps({"passed": True, "protocol": report["protocol"], "calls": len(errors)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.snapshot, args.out))

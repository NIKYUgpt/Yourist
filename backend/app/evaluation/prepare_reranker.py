"""Отдельная загрузка фиксированной ревизии весов; пользовательские тексты не отправляются."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from app.modules.search.reranker import MODEL, REVISION


def main(out: str):
    path = Path(
        snapshot_download(
            MODEL,
            revision=REVISION,
            allow_patterns=["*.json", "model.safetensors", "sentencepiece.bpe.model"],
            local_dir=out,
        )
    )
    manifest = {"model": MODEL, "revision": REVISION, "files_sha256": {}}
    for file in sorted(path.iterdir()):
        if file.is_file() and file.name != "yourist-manifest.json":
            with file.open("rb") as handle:
                manifest["files_sha256"][file.name] = hashlib.file_digest(
                    handle, "sha256"
                ).hexdigest()
    (path / "yourist-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    main(parser.parse_args().out)

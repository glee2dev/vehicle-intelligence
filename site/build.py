#!/usr/bin/env python3
"""Inline site/replay.json into site/index.template.html -> site/index.html (single file, no backend)."""
import json
from pathlib import Path
here = Path(__file__).parent
replay = json.load(open(here / "replay.json"))
payload = json.dumps(replay, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
html = open(here / "index.template.html", encoding="utf-8").read().replace("/*__REPLAY__*/", payload)
open(here / "index.html", "w", encoding="utf-8").write(html)
docs = here.parent / "docs"; docs.mkdir(exist_ok=True)
open(docs / "index.html", "w", encoding="utf-8").write(html); (docs / ".nojekyll").touch()
print(f"wrote site/index.html and docs/index.html ({len(html)/1024:.0f} KB)")

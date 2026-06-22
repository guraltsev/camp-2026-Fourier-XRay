#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse


PLUGIN_KEY = "@jupyterlite/pyodide-kernel-extension:kernel"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch this JupyterLite site for its GitHub Pages URL."
    )
    parser.add_argument(
        "--remote",
        default="origin",
        help="Git remote used to infer the GitHub Pages URL.",
    )
    site_group = parser.add_mutually_exclusive_group()
    site_group.add_argument(
        "--site-url",
        help="Explicit Pages URL. Use this for custom domains or non-origin remotes.",
    )
    site_group.add_argument(
        "--local",
        action="store_true",
        help="Patch for local preview with npm run serve.",
    )
    return parser.parse_args()


def parse_github_remote(remote_url: str) -> tuple[str, str]:
    https_match = re.match(r"^https://github\.com/([^/]+)/(.+?)(?:\.git)?/?$", remote_url)
    if https_match:
        return https_match.group(1), https_match.group(2)

    ssh_match = re.match(r"^git@github\.com:([^/]+)/(.+?)(?:\.git)?$", remote_url)
    if ssh_match:
        return ssh_match.group(1), ssh_match.group(2)

    parsed = urlparse(remote_url)
    if parsed.hostname == "github.com":
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2:
            repo = parts[1]
            if repo.endswith(".git"):
                repo = repo[:-4]
            return parts[0], repo

    raise RuntimeError(f"Cannot infer GitHub owner and repository from remote URL: {remote_url}")


def remote_url(remote: str) -> str:
    result = subprocess.run(
        ["git", "remote", "get-url", remote],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def infer_site_url(remote: str) -> str:
    owner, repo = parse_github_remote(remote_url(remote))
    owner_slug = owner.lower()
    if repo.lower() == f"{owner_slug}.github.io":
        return f"https://{owner_slug}.github.io/"
    return f"https://{owner_slug}.github.io/{repo}/"


def patch_config(config_path: Path, pyodide_url: str) -> None:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    config = (
        data.setdefault("jupyter-config-data", {})
        .setdefault("litePluginSettings", {})
        .setdefault(PLUGIN_KEY, {})
    )
    config["pyodideUrl"] = pyodide_url
    config_path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"Updated {config_path} with pyodideUrl={config['pyodideUrl']}")


def main() -> int:
    args = parse_args()
    if args.local:
        pyodide_url = "http://localhost:8000/pyodide/pyodide.js"
    else:
        site_url = args.site_url or infer_site_url(args.remote)
        pyodide_url = site_url.rstrip("/") + "/pyodide/pyodide.js"
    patch_config(Path("jupyter-lite.json"), pyodide_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

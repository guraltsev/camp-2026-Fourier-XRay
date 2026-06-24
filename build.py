#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import quote, urlparse


PLUGIN_KEY = "@jupyterlite/pyodide-kernel-extension:kernel"
LAB_URL_START = "<!-- jupyterlite-lab-url:start -->"
LAB_URL_END = "<!-- jupyterlite-lab-url:end -->"


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


def workspace_notebook_path(workspace: dict) -> str | None:
    data = workspace.get("data", {})
    if not isinstance(data, dict):
        return None

    recents = data.get("docmanager:recents", {}).get("opened", [])
    for item in recents:
        if item.get("contentType") == "notebook" and item.get("path"):
            return item["path"]

    for key, value in data.items():
        if key.startswith("notebook:") and isinstance(value, dict):
            notebook_data = value.get("data", {})
            if isinstance(notebook_data, dict) and notebook_data.get("path"):
                return notebook_data["path"]
    return None


def workspace_title(workspace_id: str, workspace: dict) -> str:
    path = workspace_notebook_path(workspace) or workspace_id
    stem = Path(path).stem
    stem = re.sub(r"^\d+[-_]*", "", stem)
    words = re.sub(r"[-_]+", " ", stem).split()
    if not words:
        return workspace_id
    return " ".join(word[:1].upper() + word[1:] for word in words)


def refresh_workspace_index(root: Path) -> dict[str, dict]:
    workspaces_dir = root / "files" / "workspaces"
    api_dir = root / "api" / "workspaces"
    api_path = api_dir / "all.json"
    workspace_index: dict[str, dict] = (
        json.loads(api_path.read_text(encoding="utf-8")) if api_path.is_file() else {}
    )

    # Deploy repositories usually already carry api/workspaces/all.json from the
    # template build. If raw workspace files are also present, refresh the API
    # from them so local edits still propagate.
    if workspaces_dir.is_dir():
        workspace_index = {}
        for path in sorted(workspaces_dir.glob("*.jupyterlab-workspace")):
            workspace = json.loads(path.read_text(encoding="utf-8"))
            metadata = workspace.setdefault("metadata", {})
            workspace_id = metadata.setdefault("id", path.stem)
            workspace_index[workspace_id] = workspace

    api_dir.mkdir(parents=True, exist_ok=True)
    (api_dir / "all.json").write_text(
        json.dumps(workspace_index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return workspace_index


def patch_readme(readme_path: Path, site_url: str, workspaces: dict[str, dict]) -> None:
    if readme_path.exists():
        original = readme_path.read_text(encoding="utf-8")
    else:
        original = ""

    lab_url = site_url + "lab/index.html"
    workspace_lines = []
    for workspace_id, workspace in sorted(workspaces.items()):
        workspace_url = site_url + "lab/index.html?workspace=" + quote(
            workspace_id, safe=""
        )
        workspace_lines.append(
            f"- [{workspace_title(workspace_id, workspace)}]({workspace_url})"
        )
    if not workspace_lines:
        workspace_lines = ["- No bundled workspaces were found."]

    block = "\n".join(
        [
            LAB_URL_START,
            "## JupyterLab",
            "",
            f"Open the notebook environment: [{lab_url}]({lab_url})",
            "",
            "Open a notebook workspace:",
            "",
            *workspace_lines,
            LAB_URL_END,
        ]
    )

    start = original.find(LAB_URL_START)
    end = original.find(LAB_URL_END)
    if start != -1 and end != -1 and start < end:
        updated = original[:start] + block + original[end + len(LAB_URL_END) :]
    else:
        updated = original.rstrip() + "\n\n" + block + "\n"

    readme_path.write_text(updated, encoding="utf-8")
    print(f"Updated {readme_path} with lab URL={lab_url}")


def main() -> int:
    args = parse_args()
    if args.local:
        site_url = "http://localhost:8000/"
    else:
        site_url = args.site_url or infer_site_url(args.remote)
    site_url = site_url.rstrip("/") + "/"
    pyodide_url = site_url + "pyodide/pyodide.js"
    workspaces = refresh_workspace_index(Path("."))
    patch_config(Path("jupyter-lite.json"), pyodide_url)
    patch_readme(Path("README.md"), site_url, workspaces)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

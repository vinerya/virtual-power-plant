"""MkDocs build hooks for the documentation site (``hooks:`` in mkdocs.yml).

The Markdown under ``docs/`` is written to read well on GitHub too, so it
links to repository files with relative paths (``../src/vpp/settings.py``,
``../CHANGELOG.md``). This module makes those links work on the site and
generates the pages that come from the code rather than from ``docs/``:

* ``changelog.md``, ``contributing.md``, ``license.md``: CHANGELOG.md,
  CONTRIBUTING.md and LICENSE from the repository root, with their links
  rewritten for the site (nothing is copied into ``docs/``);
* ``reference/openapi.json``: the OpenAPI schema of
  ``vpp.api.app.create_app()``, rendered by ``reference/rest-api.md``;
* ``reference/cli.md``: the ``--help`` text of every ``vpp`` command;
* ``reference/redoc/``: ReDoc's standalone bundle, taken from the pinned npm
  release (hash-checked, cached in ``~/.cache/vpp-docs``) so the site serves
  it itself instead of loading it from a CDN.

Links that leave ``docs/`` and are not one of the pages above point at the
file on GitHub (``main`` branch). ``<!-- include: PATH#NAME -->`` in a page
inserts the part of PATH between ``<!-- docs:NAME:start -->`` and
``<!-- docs:NAME:end -->`` (used to show the README's maturity table on the
home page without keeping a second copy).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import posixpath
import re
import tarfile
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mkdocs.structure.files import File

if TYPE_CHECKING:
    import click
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.files import Files
    from mkdocs.structure.pages import Page

REPO_URL = "https://github.com/vinerya/virtual-power-plant"
BRANCH = "main"
ROOT = Path(__file__).resolve().parent.parent

# Repository files published as site pages: repo path -> page path in the site.
ROOT_PAGES = {
    "CHANGELOG.md": "changelog.md",
    "CONTRIBUTING.md": "contributing.md",
    "LICENSE": "license.md",
}
OPENAPI_PAGE = "reference/openapi.json"
CLI_PAGE = "reference/cli.md"

# ReDoc renders openapi.json on reference/rest-api.md. Pinned npm release and
# its registry integrity hash (`npm view redoc@2.5.2 dist.integrity`).
REDOC_VERSION = "2.5.2"
REDOC_INTEGRITY = "sha512-sTJfItvRkcDTojB6wdLN4M+Ua6mlZwElV21Tf8Mn7IbQF/1Os6GvgQpZyLWPGZZHbhy7GC1Or1hTMHfz1vKh5A=="
REDOC_DIR = "reference/redoc"

# [text](target) or [text](target "title"); targets never contain spaces here.
_LINK = re.compile(r"(\]\()([^)\s]+)((?:\s+\"[^\"]*\")?\))")
_INCLUDE = re.compile(r"<!--\s*include:\s*([^#\s]+)#([\w-]+)\s*-->")
_EXTERNAL = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|//)", re.IGNORECASE)
_FENCE = re.compile(r"^\s*(```|~~~)")


def _github_url(repo_path: str) -> str:
    kind = "tree" if (ROOT / repo_path).is_dir() else "blob"
    return f"{REPO_URL}/{kind}/{BRANCH}/{repo_path}"


def _rewrite_target(target: str, source: str, page: str) -> str:
    """Map a link written in repo file ``source`` to one valid on site page ``page``."""
    if _EXTERNAL.match(target):
        return target
    path, sep, anchor = target.partition("#")
    if not path:
        if source.startswith("docs/") or source in ROOT_PAGES:
            return target  # anchor on the same page
        path = source  # anchor in an included file: link to that file
    repo_path = posixpath.normpath(posixpath.join(posixpath.dirname(source), path))
    if repo_path.startswith("docs/"):
        dest = repo_path.removeprefix("docs/")
    elif repo_path in ROOT_PAGES:
        dest = ROOT_PAGES[repo_path]
    else:
        return _github_url(repo_path) + sep + anchor
    return posixpath.relpath(dest, posixpath.dirname(page) or ".") + sep + anchor


def rewrite_links(markdown: str, source: str, page: str, *, only_outside_docs: bool) -> str:
    """Rewrite relative Markdown links outside fenced code blocks.

    ``source`` is the repo path the Markdown was written in, ``page`` the
    site page it ends up on. With ``only_outside_docs`` links that stay
    inside ``docs/`` are left for MkDocs to resolve (and validate).
    """
    out: list[str] = []
    fence: str | None = None
    for line in markdown.splitlines(keepends=True):
        match = _FENCE.match(line)
        if match:
            if fence is None:
                fence = match.group(1)
            elif match.group(1) == fence:
                fence = None
        if fence is not None or match:
            out.append(line)
            continue

        def repl(m: re.Match[str]) -> str:
            target = m.group(2)
            if only_outside_docs:
                path = target.partition("#")[0]
                resolved = posixpath.normpath(posixpath.join(posixpath.dirname(source), path))
                if not path or _EXTERNAL.match(target) or resolved.startswith("docs/"):
                    return m.group(0)
            return m.group(1) + _rewrite_target(target, source, page) + m.group(3)

        out.append(_LINK.sub(repl, line))
    return "".join(out)


def _include(markdown: str, page: str) -> str:
    def repl(m: re.Match[str]) -> str:
        source, name = m.group(1), m.group(2)
        text = (ROOT / source).read_text(encoding="utf-8")
        start, end = f"<!-- docs:{name}:start -->", f"<!-- docs:{name}:end -->"
        if start not in text or end not in text:
            raise ValueError(f"{source} has no {start} ... {end} section")
        section = text.split(start, 1)[1].split(end, 1)[0].strip("\n")
        return rewrite_links(section, source, page, only_outside_docs=False)

    return _INCLUDE.sub(repl, markdown)


def _root_page(source: str, page: str) -> str:
    text = (ROOT / source).read_text(encoding="utf-8")
    if not source.endswith(".md"):
        title = source.title()
        return f"# {title}\n\n```text\n{text.rstrip()}\n```\n"
    return rewrite_links(text, source, page, only_outside_docs=False)


def _redoc_files() -> dict[str, bytes]:
    """ReDoc's standalone bundle from the pinned npm tarball, checked against its hash.

    Served from the site itself, so the REST API page makes no third-party
    requests. The tarball is cached in ``~/.cache/vpp-docs``.
    """
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "vpp-docs"
    tarball = cache / f"redoc-{REDOC_VERSION}.tgz"

    def verified(data: bytes) -> bool:
        digest = base64.b64encode(hashlib.sha512(data).digest()).decode()
        return f"sha512-{digest}" == REDOC_INTEGRITY

    data = tarball.read_bytes() if tarball.is_file() else b""
    if not verified(data):
        url = f"https://registry.npmjs.org/redoc/-/redoc-{REDOC_VERSION}.tgz"
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        if not verified(data):
            raise RuntimeError(f"{url} does not match the pinned integrity hash")
        cache.mkdir(parents=True, exist_ok=True)
        tarball.write_bytes(data)
    members = {
        "package/bundles/redoc.standalone.js": "redoc.standalone.js",
        "package/bundles/redoc.standalone.js.LICENSE.txt": "redoc.standalone.js.LICENSE.txt",
        "package/LICENSE": "LICENSE.txt",
    }
    out: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member, name in members.items():
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"{member} missing from redoc-{REDOC_VERSION}.tgz")
            out[f"{REDOC_DIR}/{name}"] = extracted.read()
    # The side menu shows a "redocly logo" image loaded from cdn.redoc.ly;
    # serve an empty inline SVG instead so the page stays first-party only
    # (extra.css hides the element).
    bundle = f"{REDOC_DIR}/redoc.standalone.js"
    logo = b'src:"https://cdn.redoc.ly/redoc/logo-mini.svg"'
    if out[bundle].count(logo) != 1:
        raise RuntimeError(f"redoc {REDOC_VERSION}: logo URL not found once; update the patch")
    empty_svg = b"src:\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg'/%3E\""
    out[bundle] = out[bundle].replace(logo, empty_svg)
    return out


def _openapi_json() -> str:
    from vpp.api.app import create_app

    return json.dumps(create_app().openapi(), indent=2, sort_keys=False) + "\n"


def _cli_markdown() -> str:
    import click

    from vpp.cli.main import cli

    lines = [
        "# CLI reference",
        "",
        "Generated at build time from `vpp --help` and the help of every",
        "subcommand. Every command also takes `--help`.",
        "",
    ]

    def walk(cmd: click.Command, name: str, parent: click.Context | None) -> None:
        ctx = click.Context(
            cmd, info_name=name, parent=parent, terminal_width=88, max_content_width=88
        )
        if name == "sunspec":
            # Delegates its arguments to an argparse parser.
            from vpp.simulators.sunspec import build_parser

            old = os.environ.get("COLUMNS")
            os.environ["COLUMNS"] = "88"
            try:
                text = build_parser(prog=ctx.command_path).format_help()
            finally:
                if old is None:
                    os.environ.pop("COLUMNS", None)
                else:
                    os.environ["COLUMNS"] = old
        else:
            text = cmd.get_help(ctx)
        depth = min(ctx.command_path.count(" ") + 1, 3)
        lines.extend([f"{'#' * (depth + 1)} `{ctx.command_path}`", "", "```text"])
        lines.extend([text.rstrip(), "```", ""])
        if isinstance(cmd, click.Group):
            for sub_name in cmd.list_commands(ctx):
                sub = cmd.get_command(ctx, sub_name)
                if sub is not None and not sub.hidden:
                    walk(sub, sub_name, ctx)

    walk(cli, "vpp", None)
    return "\n".join(lines)


def on_files(files: Files, config: MkDocsConfig) -> Files:
    for source, page in ROOT_PAGES.items():
        files.append(File.generated(config, page, content=_root_page(source, page)))
    files.append(File.generated(config, OPENAPI_PAGE, content=_openapi_json()))
    files.append(File.generated(config, CLI_PAGE, content=_cli_markdown()))
    for path, content in _redoc_files().items():
        files.append(File.generated(config, path, content=content))
    return files


def on_page_markdown(
    markdown: str, page: Page, config: MkDocsConfig, files: Files, **kwargs: Any
) -> str:
    if page.file.generated_by is not None:
        # Links were rewritten when the page was generated. "Edit this page"
        # points at the repository file the page came from, if there is one.
        sources = {page_path: source for source, page_path in ROOT_PAGES.items()}
        source = sources.get(page.file.src_uri)
        page.edit_url = f"{REPO_URL}/edit/{BRANCH}/{source}" if source else None
        return markdown
    page_path = page.file.src_uri
    markdown = _include(markdown, page_path)
    return rewrite_links(markdown, f"docs/{page_path}", page_path, only_outside_docs=True)

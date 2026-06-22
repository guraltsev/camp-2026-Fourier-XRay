"""Convert toolkit Markdown and HTML fragments into frontend payload blocks."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "MARKDOWN_RENDERER_ESM",
    "RenderBlock",
    "html_block",
    "markdown_block",
    "render_markdown_payload",
    "stdout_block",
]


# Shared anywidget ESM for toolkit-authored Markdown fragments. The fallback is
# intentionally plain escaped Markdown so unsupported hosts do not see a partial
# Markdown dialect masquerading as complete rendering.
MARKDOWN_RENDERER_ESM = r"""
function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function markdownRenderer() {
  if (typeof globalThis.mathToolkitRenderMarkdown === "function") {
    return globalThis.mathToolkitRenderMarkdown;
  }
  return null;
}

function typesetMath(element) {
  if (typeof globalThis.renderMathInElement === "function") {
    globalThis.renderMathInElement(element, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "$", right: "$", display: false },
        { left: "\\(", right: "\\)", display: false },
      ],
      output: "html",
      throwOnError: false,
    });
    return;
  }

  const mathJax = globalThis.MathJax;
  if (typeof mathJax?.typesetPromise === "function") {
    mathJax.typesetPromise([element]).catch((error) => {
      console.warn("[math_toolkit] MathJax failed to typeset Markdown.", error);
    });
    return;
  }
  if (typeof mathJax?.typeset === "function") {
    mathJax.typeset([element]);
  }
}

function renderMarkdownArea(element, payload) {
  const block = typeof payload === "string" ? { kind: "markdown", text: payload } : (payload || {});
  element.classList.add("mt-markdown");
  if (block.kind === "html") {
    element.style.removeProperty("white-space");
    element.innerHTML = block.html || "";
    typesetMath(element);
    return;
  }
  const text = block.text || block.markdown || "";
  if (block.rendered_html) {
    element.style.removeProperty("white-space");
    element.innerHTML = block.rendered_html;
    typesetMath(element);
    return;
  }
  const render = markdownRenderer();
  if (render) {
    try {
      element.style.removeProperty("white-space");
      element.innerHTML = render(text);
      typesetMath(element);
      return;
    } catch (error) {
      console.warn("[math_toolkit] Host Markdown renderer failed.", error);
    }
  }
  element.style.whiteSpace = "pre-wrap";
  element.textContent = text;
}

function blockElement(block) {
  if (block.kind === "stdout") {
    const pre = document.createElement("pre");
    pre.className = "mt-stdout";
    pre.textContent = block.text || "";
    return pre;
  }
  const div = document.createElement("div");
  div.className = "mt-markdown";
  renderMarkdownArea(div, block);
  return div;
}
"""


@dataclass(frozen=True)
class RenderBlock:
    """Describe one frontend-renderable text fragment."""

    kind: str
    text: str = ""
    html: str = ""
    rendered_html: str = ""

    def to_payload(self) -> dict[str, str]:
        """Return a JSON-compatible block payload."""

        if self.kind == "html":
            return {"kind": "html", "html": self.html}
        payload = {"kind": self.kind, "text": self.text}
        if self.rendered_html:
            payload["rendered_html"] = self.rendered_html
        return payload


def markdown_block(markdown: str, *, rendered_html: str = "") -> dict[str, str]:
    """Return a Markdown render block payload."""

    return RenderBlock(
        "markdown",
        text=str(markdown),
        rendered_html=str(rendered_html),
    ).to_payload()


def render_markdown_payload(
    markdown: str,
    *,
    rendered_html: str = "",
) -> dict[str, str]:
    """Return a Markdown payload with optional host-rendered HTML."""

    return markdown_block(str(markdown), rendered_html=str(rendered_html))


def html_block(html: str) -> dict[str, str]:
    """Return a trusted HTML render block payload."""

    return RenderBlock("html", html=str(html)).to_payload()


def stdout_block(text: str) -> dict[str, str]:
    """Return a stdout render block payload."""

    return RenderBlock("stdout", text=str(text)).to_payload()

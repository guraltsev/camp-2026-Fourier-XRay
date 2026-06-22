"""Provide small anywidget-backed primitives for toolkit plotting controls."""

from __future__ import annotations

from collections.abc import Callable
import math

import traitlets
from ipywidgets.widgets.widget import Widget, widget_serialization

from .child_mount import CHILD_MOUNT_ESM
from .markdown import MARKDOWN_RENDERER_ESM, render_markdown_payload

__all__ = [
    "ButtonWidget",
    "HBoxWidget",
    "MarkdownWidget",
    "SliderWidget",
    "TextEntryWidget",
    "VBoxWidget",
]


def _widget_base_class() -> type[object]:
    """Return the installed anywidget base class or raise a clear error."""

    try:
        import anywidget
    except ImportError as exc:
        from ..errors import PlotSpecError

        raise PlotSpecError(
            "backend='anywidget' requires the anywidget package to be installed."
        ) from exc
    return anywidget.AnyWidget


class BoxWidget(_widget_base_class()):
    """Arrange child widgets in a small flex container."""

    children = traitlets.List(
        trait=traitlets.Instance(Widget),
        default_value=[],
    ).tag(sync=True, **widget_serialization)
    direction = traitlets.Unicode("row").tag(sync=True)
    gap = traitlets.Unicode("0.35rem").tag(sync=True)
    align_items = traitlets.Unicode("center").tag(sync=True)
    class_name = traitlets.Unicode("").tag(sync=True)
    child_mount_policy = traitlets.Dict(default_value={}).tag(sync=True)
    _css = r"""
.mt-widget-box {
  box-sizing: border-box;
  display: flex;
  min-width: 0;
}
.mt-widget-box > * {
  min-width: 0;
}
.mt-widget-box.mt-param-row {
  display: flex;
  gap: 0;
  width: 100%;
}
.mt-param-row__label-value {
  flex: 0 0 auto;
  justify-content: flex-start;
  max-width: 12rem;
  min-width: 0;
}
.mt-param-row__spacer {
  flex: 1 1 auto;
  min-width: 0;
}
.mt-param-row__label {
  flex: 0 1 auto;
  max-width: 4.5rem;
  min-width: 0;
  overflow: hidden;
  white-space: nowrap;
}
.mt-param-row__label.mt-markdown--overflowing {
  -webkit-mask-image: linear-gradient(to right, #000 calc(100% - 1.15rem), transparent);
  mask-image: linear-gradient(to right, #000 calc(100% - 1.15rem), transparent);
}
.mt-param-row__label .mt-markdown,
.mt-param-row__label p {
  overflow: hidden;
  text-overflow: clip;
  white-space: nowrap;
}
.mt-param-row__slider-stack {
  flex: 0 1 auto;
  justify-content: flex-start;
  min-width: calc(10ch + 4rem + 0.36rem);
}
.mt-param-row__slider-stack > *:nth-child(2) {
  flex: 1 1 4rem;
  max-width: 12rem;
  min-width: 4rem;
}
.mt-param-row__edit {
  flex: 0 0 1.3rem;
  margin-left: 1em;
}
.mt-param-row__reset {
  flex: 0 0 1.15rem;
}
.mt-plot-shell[data-layout="compact"] .mt-param-row {
  flex-wrap: wrap;
  gap: 0.18rem 0.25rem;
  overflow: visible;
}
.mt-plot-shell[data-layout="compact"] .mt-param-row__spacer {
  display: none;
}
.mt-plot-shell[data-layout="compact"] .mt-param-row__label-value {
  flex: 0 1 auto;
}
.mt-plot-shell[data-layout="compact"] .mt-param-row__slider-stack {
  flex: 1 1 calc(10ch + 4rem + 0.36rem);
}
.mt-plot-shell[data-layout="compact"] .mt-param-row__edit {
  margin-left: 0;
}
"""

    _esm = CHILD_MOUNT_ESM + r"""
function render({ model, el }) {
  el.classList.add("mt-widget-box");

  let childViews = [];
  let renderToken = 0;

  function syncLayout() {
    el.style.flexDirection = model.get("direction");
    el.style.gap = model.get("gap");
    el.style.alignItems = model.get("align_items");
    el.className = `mt-widget-box ${model.get("class_name") || ""}`.trim();
  }

  async function renderChildren() {
    const token = ++renderToken;
    const previousViews = childViews;
    childViews = [];
    el.replaceChildren();
    previousViews.forEach(disposeView);

    const children = model.get("children") || [];
    for (let index = 0; index < children.length; index += 1) {
      const view = await createWidgetView(model, children[index]);
      if (token !== renderToken) {
        disposeView(view);
        return;
      }
      childViews.push(view);
      if (view && view.el) {
        el.append(view.el);
      }
    }
  }

  for (const name of ["direction", "gap", "align_items", "class_name"]) {
    model.on(`change:${name}`, syncLayout);
  }
  model.on("change:children", renderChildren);
  syncLayout();
  renderChildren();

  return () => {
    renderToken += 1;
    for (const name of ["direction", "gap", "align_items", "class_name"]) {
      model.off(`change:${name}`, syncLayout);
    }
    model.off("change:children", renderChildren);
    childViews.forEach(disposeView);
    childViews = [];
  };
}

export default { render };
"""

    def __init__(
        self,
        children: tuple[Widget, ...] = (),
        *,
        direction: str = "row",
        gap: str = "0.35rem",
        align_items: str = "center",
        class_name: str = "",
        child_mount_policy: dict[str, object] | None = None,
    ) -> None:
        """Create a flex container for child widgets."""

        super().__init__()
        self.children = list(children)
        self.direction = direction
        self.gap = gap
        self.align_items = align_items
        self.class_name = class_name
        self.child_mount_policy = (
            {} if child_mount_policy is None else dict(child_mount_policy)
        )


class HBoxWidget(BoxWidget):
    """Arrange child widgets horizontally."""

    def __init__(
        self,
        children: tuple[Widget, ...] = (),
        *,
        gap: str = "0.35rem",
        align_items: str = "center",
        class_name: str = "",
    ) -> None:
        """Create a horizontal flex container."""

        super().__init__(
            children,
            direction="row",
            gap=gap,
            align_items=align_items,
            class_name=class_name,
        )


class VBoxWidget(BoxWidget):
    """Arrange child widgets vertically."""

    def __init__(
        self,
        children: tuple[Widget, ...] = (),
        *,
        gap: str = "0.25rem",
        align_items: str = "stretch",
        class_name: str = "",
    ) -> None:
        """Create a vertical flex container."""

        super().__init__(
            children,
            direction="column",
            gap=gap,
            align_items=align_items,
            class_name=class_name,
        )


class MarkdownWidget(_widget_base_class()):
    """Render a small Markdown label fragment."""

    value = traitlets.Unicode("").tag(sync=True)
    rendered_html = traitlets.Unicode("").tag(sync=True)
    class_name = traitlets.Unicode("").tag(sync=True)

    _css = r"""
.mt-markdown {
  min-width: 0;
}
.mt-markdown p {
  margin: 0;
}
"""

    _esm = MARKDOWN_RENDERER_ESM + r"""
function render({ model, el }) {
  function syncTitle() {
    const overflowing = el.scrollWidth > el.clientWidth;
    el.classList.toggle("mt-markdown--overflowing", overflowing);
    if (overflowing) {
      el.title = model.get("value") || "";
    } else {
      el.removeAttribute("title");
    }
  }

  function sync() {
    el.className = `mt-markdown ${model.get("class_name") || ""}`.trim();
    renderMarkdownArea(el, {
      kind: "markdown",
      text: model.get("value") || "",
      rendered_html: model.get("rendered_html") || "",
    });
    requestAnimationFrame(syncTitle);
  }
  model.on("change:value", sync);
  model.on("change:rendered_html", sync);
  model.on("change:class_name", sync);
  window.addEventListener("resize", syncTitle);
  sync();
  return () => {
    model.off("change:value", sync);
    model.off("change:rendered_html", sync);
    model.off("change:class_name", sync);
    window.removeEventListener("resize", syncTitle);
  };
}

export default { render };
"""

    def __init__(
        self,
        value: str = "",
        *,
        class_name: str = "",
        markdown_payload: Callable[[str], dict[str, str]] | None = None,
    ) -> None:
        """Create a Markdown label widget."""

        super().__init__()
        self.class_name = class_name
        self._markdown_payload = markdown_payload or render_markdown_payload
        self.set_markdown(value)

    def set_markdown(self, value: str) -> None:
        """Update Markdown text and any host-rendered HTML payload."""

        payload = self._markdown_payload(value)
        self.value = payload["text"]
        self.rendered_html = payload.get("rendered_html", "")


class TextEntryWidget(_widget_base_class()):
    """Render a text entry with a commit counter."""

    value = traitlets.Unicode("").tag(sync=True)
    disabled = traitlets.Bool(False).tag(sync=True)
    commit_count = traitlets.Int(0).tag(sync=True)
    class_name = traitlets.Unicode("").tag(sync=True)
    style = traitlets.Dict(default_value={}).tag(sync=True)

    _css = r"""
.mt-text-entry {
  background: transparent;
  border: 1px solid #d8dee6;
  border-radius: 0.2rem;
  box-sizing: border-box;
  font: inherit;
  min-width: 0;
  outline: 0;
  padding: 0.12rem 0.25rem;
}
.mt-text-entry:focus {
  outline: 1px solid rgba(71, 85, 105, 0.55);
  outline-offset: 0;
}
.mt-text-entry--value {
  background: #f8fafc;
  font-size: 0.72rem;
  text-align: right;
  width: 7ch;
}
.mt-text-entry--limit {
  background: transparent;
  border: 0;
  box-shadow: none;
  font-size: 0.68rem;
  height: 0.95rem;
  -webkit-mask-image: linear-gradient(to right, transparent 0, #000 0.55rem);
  mask-image: linear-gradient(to right, transparent 0, #000 0.55rem);
  padding: 0;
  width: 5ch;
}
.mt-text-entry--limit:focus {
  outline: 1px solid rgba(71, 85, 105, 0.45);
}
.mt-text-entry--minimum {
  text-align: right;
}
.mt-text-entry--maximum {
  text-align: left;
}
"""

    _esm = r"""
function render({ model, el }) {
  const input = document.createElement("input");
  input.type = "text";
  el.replaceChildren(input);

  function applyStyle() {
    input.style.cssText = "";
    const style = model.get("style") || {};
    for (const [name, value] of Object.entries(style)) {
      const property = name.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`);
      if (value == null || value === "") {
        input.style.removeProperty(property);
      } else {
        input.style.setProperty(property, String(value));
      }
    }
  }

  function sync() {
    input.className = `mt-text-entry ${model.get("class_name") || ""}`.trim();
    if (document.activeElement !== input) {
      input.value = model.get("value") || "";
    }
    input.disabled = Boolean(model.get("disabled"));
    applyStyle();
  }

  input.addEventListener("change", () => {
    model.set("value", input.value);
    model.set("commit_count", model.get("commit_count") + 1);
    model.save_changes();
  });

  for (const name of ["value", "disabled", "class_name", "style"]) {
    model.on(`change:${name}`, sync);
  }
  sync();

  return () => {
    for (const name of ["value", "disabled", "class_name", "style"]) {
      model.off(`change:${name}`, sync);
    }
  };
}

export default { render };
"""

    def __init__(
        self,
        value: str = "",
        *,
        disabled: bool = False,
        class_name: str = "",
        style: dict[str, str] | None = None,
    ) -> None:
        """Create a text entry widget."""

        super().__init__()
        self.value = value
        self.disabled = disabled
        self.class_name = class_name
        self.style = {} if style is None else dict(style)


class SliderWidget(_widget_base_class()):
    """Render a range slider with traitlet-backed value sync."""

    value = traitlets.Float(0.0).tag(sync=True)
    browser_value = traitlets.Float(0.0)
    minimum = traitlets.Float(0.0).tag(sync=True)
    maximum = traitlets.Float(1.0).tag(sync=True)
    step = traitlets.Float(0.01).tag(sync=True)
    disabled = traitlets.Bool(False).tag(sync=True)
    release_count = traitlets.Int(0).tag(sync=True)
    class_name = traitlets.Unicode("").tag(sync=True)

    _css = r"""
.mt-slider {
  box-sizing: border-box;
  height: 0.8rem;
  margin: 0;
  min-width: 0;
  max-width: 12rem;
  width: 100%;
}
"""

    _esm = r"""
function render({ model, el }) {
  const input = document.createElement("input");
  input.type = "range";
  el.replaceChildren(input);

  let isInteracting = false;
  let isKeyboardInteracting = false;
  let isPointerInteracting = false;
  let pendingRelease = false;
  let releaseTimer = 0;
  let previewTimer = 0;
  let pendingPreviewValue = null;
  let previewInFlight = false;
  let previewInFlightSeq = null;
  let lastPreviewSentAt = 0;
  let lastChangeValue = null;
  let awaitingModelValue = null;
  let messageSeq = 0;
  const previewIntervalMs = 50;

  const sliderKeys = new Set([
    "ArrowLeft",
    "ArrowRight",
    "ArrowUp",
    "ArrowDown",
    "Home",
    "End",
    "PageUp",
    "PageDown",
  ]);

  function sync() {
    input.className = `mt-slider ${model.get("class_name") || ""}`.trim();
    input.min = model.get("minimum");
    input.max = model.get("maximum");
    input.step = model.get("step");
    syncValueFromModel();
    input.disabled = Boolean(model.get("disabled"));
  }

  function syncValueFromModel() {
    const modelValue = model.get("value");
    if (awaitingModelValue !== null) {
      if (sameSliderValue(modelValue, awaitingModelValue)) {
        awaitingModelValue = null;
        if (!sameSliderValue(input.value, modelValue)) {
          input.value = modelValue;
        }
      }
      return;
    }
    if (!isInteracting && !sameSliderValue(input.value, modelValue)) {
      input.value = modelValue;
    }
  }

  function syncDisabled() {
    input.disabled = Boolean(model.get("disabled"));
  }

  function sameSliderValue(left, right) {
    const leftNumber = Number(left);
    const rightNumber = Number(right);
    if (!Number.isFinite(leftNumber) || !Number.isFinite(rightNumber)) {
      return String(left) === String(right);
    }
    return Math.abs(leftNumber - rightNumber) <= 1e-12;
  }

  function beginInteraction() {
    isInteracting = true;
    if (releaseTimer) {
      clearTimeout(releaseTimer);
      releaseTimer = 0;
    }
  }

  function sendInput() {
    pendingRelease = true;
    pendingPreviewValue = Number(input.value);
    if (previewInFlight) {
      return;
    }
    schedulePreview();
  }

  function schedulePreview() {
    if (previewTimer || previewInFlight || pendingPreviewValue === null) {
      return;
    }
    const now = performance.now();
    const delay = Math.max(0, previewIntervalMs - (now - lastPreviewSentAt));
    previewTimer = setTimeout(() => {
      previewTimer = 0;
      if (previewInFlight || pendingPreviewValue === null) {
        return;
      }
      const value = pendingPreviewValue;
      const seq = ++messageSeq;
      pendingPreviewValue = null;
      previewInFlight = true;
      previewInFlightSeq = seq;
      lastPreviewSentAt = performance.now();
      model.send({ type: "input", value, seq });
    }, delay);
  }

  function sendRelease() {
    if (!pendingRelease && lastChangeValue === input.value) {
      return;
    }
    if (previewTimer) {
      clearTimeout(previewTimer);
      previewTimer = 0;
    }
    pendingPreviewValue = null;
    pendingRelease = false;
    lastChangeValue = input.value;
    awaitingModelValue = Number(input.value);
    lastPreviewSentAt = performance.now();
    model.send({ type: "change", value: Number(input.value), seq: ++messageSeq });
  }

  function handleCustomMessage(message) {
    if (!message || message.type !== "slider_ack") {
      return;
    }
    if (message.seq !== previewInFlightSeq) {
      return;
    }
    previewInFlight = false;
    previewInFlightSeq = null;
    schedulePreview();
  }

  function finishInteraction({ sendChange = true } = {}) {
    if (sendChange) {
      sendRelease();
    }
    if (releaseTimer) {
      clearTimeout(releaseTimer);
    }
    releaseTimer = setTimeout(() => {
      isInteracting = false;
      isKeyboardInteracting = false;
      isPointerInteracting = false;
      releaseTimer = 0;
      syncValueFromModel();
    }, 180);
  }

  input.addEventListener("pointerdown", () => {
    isPointerInteracting = true;
    beginInteraction();
  });
  input.addEventListener("pointerup", () => {
    isPointerInteracting = false;
    finishInteraction();
  });
  input.addEventListener("blur", () => {
    isKeyboardInteracting = false;
    isPointerInteracting = false;
    finishInteraction();
  });
  input.addEventListener("keydown", (event) => {
    if (!sliderKeys.has(event.key)) {
      return;
    }
    event.stopPropagation();
    isKeyboardInteracting = true;
    beginInteraction();
  });
  input.addEventListener("keyup", (event) => {
    if (!sliderKeys.has(event.key)) {
      return;
    }
    event.stopPropagation();
    isKeyboardInteracting = false;
    finishInteraction();
  });
  input.addEventListener("input", () => {
    beginInteraction();
    sendInput();
  });
  input.addEventListener("change", () => {
    if (isKeyboardInteracting || isPointerInteracting) {
      return;
    }
    finishInteraction();
  });

  model.on("change:value", syncValueFromModel);
  model.on("msg:custom", handleCustomMessage);
  model.on("change:disabled", syncDisabled);
  for (const name of ["minimum", "maximum", "step", "class_name"]) {
    model.on(`change:${name}`, sync);
  }
  sync();

  return () => {
    if (previewTimer) {
      clearTimeout(previewTimer);
      previewTimer = 0;
    }
    if (releaseTimer) {
      clearTimeout(releaseTimer);
      releaseTimer = 0;
    }
    model.off("change:value", syncValueFromModel);
    model.off("msg:custom", handleCustomMessage);
    model.off("change:disabled", syncDisabled);
    for (const name of ["minimum", "maximum", "step", "class_name"]) {
      model.off(`change:${name}`, sync);
    }
  };
}

export default { render };
"""

    def __init__(
        self,
        *,
        value: float,
        minimum: float,
        maximum: float,
        step: float,
        disabled: bool = False,
        class_name: str = "",
    ) -> None:
        """Create a range slider widget."""

        super().__init__()
        self.value = float(value)
        self.browser_value = float(value)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.step = float(step)
        self.disabled = disabled
        self.class_name = class_name
        self.on_msg(self._handle_message)

    def _handle_message(
        self,
        widget: object,
        content: dict[str, object],
        buffers: object,
    ) -> None:
        """Apply browser slider events without browser-originated trait sync."""

        message_type = content.get("type")
        if message_type not in {"input", "change"}:
            return
        try:
            value = float(content.get("value"))
        except (TypeError, ValueError):
            return
        if not math.isfinite(value):
            return
        self.browser_value = value
        sequence = content.get("seq")
        if message_type == "input" and isinstance(sequence, int):
            self.send({"type": "slider_ack", "seq": sequence})
        if message_type == "change":
            self.release_count += 1


class ButtonWidget(_widget_base_class()):
    """Render a button with a click counter."""

    label = traitlets.Unicode("Button").tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    disabled = traitlets.Bool(False).tag(sync=True)
    click_count = traitlets.Int(0).tag(sync=True)
    class_name = traitlets.Unicode("").tag(sync=True)

    _css = r"""
.mt-button {
  align-items: center;
  background: transparent;
  border: 0;
  border-radius: 0.2rem;
  cursor: pointer;
  display: inline-flex;
  font: inherit;
  height: 1.3rem;
  justify-content: center;
  padding: 0;
  width: 1.3rem;
}
.mt-button svg {
  height: 1rem;
  width: 1rem;
}
.mt-button--reset svg {
  height: 0.85rem;
  width: 0.85rem;
}
.mt-button:hover {
  background: #eef2f7;
}
"""

    _esm = r"""
function gearSvg() {
  return `
    <svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Z"></path>
      <path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.6V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1A2 2 0 1 1 4.2 17l.1-.1a1.7 1.7 0 0 0 .3-1.9 1.7 1.7 0 0 0-1.6-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9L4.3 7A2 2 0 1 1 7 4.2l.1.1a1.7 1.7 0 0 0 1.9.3 1.7 1.7 0 0 0 1-1.6V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.6 1.7 1.7 0 0 0 1.9-.3l.1-.1A2 2 0 1 1 19.8 7l-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.6 1h.1a2 2 0 1 1 0 4H21a1.7 1.7 0 0 0-1.6 1Z"></path>
    </svg>`;
}

function refreshSvg() {
  return `
    <svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M21 12a9 9 0 1 1-2.64-6.36"></path>
      <path d="M21 3v6h-6"></path>
    </svg>`;
}

function render({ model, el }) {
  const button = document.createElement("button");
  button.type = "button";
  el.replaceChildren(button);

  function sync() {
    const label = model.get("label") || "";
    button.className = `mt-button ${model.get("class_name") || ""}`.trim();
    if (label === "gear") {
      button.innerHTML = gearSvg();
      button.setAttribute("aria-label", model.get("title") || "Edit");
    } else if (label === "refresh") {
      button.innerHTML = refreshSvg();
      button.classList.add("mt-button--reset");
      button.setAttribute("aria-label", model.get("title") || "Reset");
    } else {
      button.textContent = label;
      button.removeAttribute("aria-label");
    }
    button.title = model.get("title") || "";
    button.disabled = Boolean(model.get("disabled"));
  }

  button.addEventListener("click", () => {
    model.set("click_count", model.get("click_count") + 1);
    model.save_changes();
  });

  for (const name of ["label", "title", "disabled", "class_name"]) {
    model.on(`change:${name}`, sync);
  }
  sync();

  return () => {
    for (const name of ["label", "title", "disabled", "class_name"]) {
      model.off(`change:${name}`, sync);
    }
  };
}

export default { render };
"""

    def __init__(
        self,
        label: str = "Button",
        *,
        title: str = "",
        disabled: bool = False,
        class_name: str = "",
    ) -> None:
        """Create a button widget."""

        super().__init__()
        self.label = label
        self.title = title
        self.disabled = disabled
        self.class_name = class_name

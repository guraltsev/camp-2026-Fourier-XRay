"""Render a plotting generation shell as one plain anywidget."""

from __future__ import annotations

import traitlets
from ipywidgets.widgets.widget import Widget, widget_serialization

from .child_mount import CHILD_MOUNT_ESM
from .markdown import MARKDOWN_RENDERER_ESM

__all__ = ["FigureShellWidget"]


def _shell_base_class() -> type[object]:
    """Return the installed anywidget base class or raise a clear error."""

    try:
        import anywidget
    except ImportError as exc:
        from ..errors import PlotSpecError

        raise PlotSpecError(
            "backend='anywidget' requires the anywidget package to be installed."
        ) from exc
    return anywidget.AnyWidget


class FigureShellWidget(_shell_base_class()):
    """Own anywidget UI state for one figure display generation."""

    generation_id = traitlets.Int().tag(sync=True)
    plot_widget = traitlets.Instance(
        Widget,
        allow_none=True,
        default_value=None,
    ).tag(sync=True, **widget_serialization)
    runtime_widgets = traitlets.List(
        trait=traitlets.Instance(Widget),
        default_value=[],
    ).tag(sync=True, **widget_serialization)
    control_widgets = traitlets.List(
        trait=traitlets.Instance(Widget),
        default_value=[],
    ).tag(sync=True, **widget_serialization)
    state = traitlets.Unicode("active").tag(sync=True)
    legend = traitlets.List(trait=traitlets.Dict(), default_value=[]).tag(sync=True)
    legend_label_widgets = traitlets.List(
        trait=traitlets.Instance(Widget),
        default_value=[],
    ).tag(sync=True, **widget_serialization)
    info = traitlets.List(trait=traitlets.Dict(), default_value=[]).tag(sync=True)
    info_markdown_widgets = traitlets.List(
        trait=traitlets.Instance(Widget),
        default_value=[],
    ).tag(sync=True, **widget_serialization)
    status = traitlets.Dict(default_value={}).tag(sync=True)
    output_items = traitlets.List(trait=traitlets.Dict(), default_value=[]).tag(sync=True)
    modal = traitlets.Dict(default_value={"state": "closed"}).tag(sync=True)
    host_name = traitlets.Unicode("jupyter").tag(sync=True)
    root_class = traitlets.Unicode("mt-host-jupyter").tag(sync=True)
    child_mount_policy = traitlets.Dict(default_value={}).tag(sync=True)
    markdown_policy = traitlets.Dict(default_value={}).tag(sync=True)

    _css = r"""
.mt-plot-shell,
.mt-plot-shell * {
  box-sizing: border-box;
}
.mt-plot-shell {
  color: #18202a;
  display: grid;
  font: 0.86rem/1.35 system-ui, sans-serif;
  gap: 0.7rem;
  grid-template-columns: minmax(0, 1fr) minmax(14rem, 22rem);
  margin-top: 0.5rem;
  position: relative;
  width: 100%;
}
.mt-plot-shell .modebar-container {
  margin: -0.75rem;
  padding: 0.75rem;
}
.mt-plot-shell .modebar {
  opacity: 0;
  transition: opacity 140ms ease;
}
.mt-plot-shell .modebar-container:hover .modebar,
.mt-plot-shell .modebar:hover,
.mt-plot-shell .modebar:focus-within {
  opacity: 1;
}
.mt-plot-shell__plot {
  min-height: 24rem;
  width: 100%;
}
.mt-plot-shell__plot-widget-view,
.mt-plot-shell__plot-widget-view > * {
  display: block;
  min-height: 24rem;
  width: 100%;
}
.mt-plot-shell__plot > .widget-box,
.mt-plot-shell__plot > .jupyter-widgets,
.mt-plot-shell__plot .js-plotly-plot,
.mt-plot-shell__plot .plot-container,
.mt-plot-shell__plot .svg-container {
  max-width: 100% !important;
  width: 100% !important;
}
.mt-plot-shell__runtime,
.mt-plot-shell__runtime * {
  display: none !important;
  height: 0 !important;
  min-height: 0 !important;
  width: 0 !important;
}
.mt-plot-shell__main,
.mt-plot-shell__side,
.mt-plot-shell__section,
.mt-plot-shell__output {
  min-width: 0;
}
.mt-plot-shell__side {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 24rem;
  min-width: 14rem;
}
.mt-plot-shell__side,
.mt-plot-shell__section {
  border: 0;
  box-shadow: none;
}
.mt-plot-shell__section[hidden],
.mt-plot-shell__output[hidden] {
  display: none;
}
.mt-plot-shell__section {
  display: flex;
  flex: 0 0 auto;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
}
.mt-plot-shell__section[data-section="label"] {
  flex-basis: var(--mt-label-section-height, 7rem);
  min-height: 4rem;
}
.mt-plot-shell__section[data-section="parameter"] {
  flex-basis: var(--mt-parameter-section-height, 8rem);
  min-height: 4.5rem;
}
.mt-plot-shell__section[data-section="info"] {
  flex-basis: var(--mt-info-section-height, 9rem);
  min-height: 5rem;
}
.mt-plot-shell:not([data-layout="compact"]) .mt-plot-shell__section[data-section="label"] {
  flex: 1 1 var(--mt-label-section-height, 7rem);
}
.mt-plot-shell:not([data-layout="compact"]) .mt-plot-shell__section[data-section="parameter"] {
  flex: 1 1 var(--mt-parameter-section-height, 8rem);
}
.mt-plot-shell:not([data-layout="compact"]) .mt-plot-shell__section[data-section="info"] {
  flex: 1 1 var(--mt-info-section-height, 9rem);
}
.mt-plot-shell__section-body {
  direction: rtl;
  flex: 1 1 auto;
  min-height: 0;
  overflow-x: hidden;
  overflow-y: auto;
  scrollbar-gutter: stable;
}
.mt-plot-shell__section-body > * {
  direction: ltr;
}
.mt-plot-shell__section-resizer {
  border-top: 1px solid rgba(148, 163, 184, 0.28);
  cursor: row-resize;
  flex: 0 0 0.45rem;
  margin: 0.04rem 0;
  min-height: 0.45rem;
  position: relative;
}
.mt-plot-shell__section-resizer::before {
  background: rgba(148, 163, 184, 0.16);
  content: "";
  height: 1px;
  left: 0;
  position: absolute;
  right: 0;
  top: 0.22rem;
}
.mt-plot-shell__section-resizer:hover::before,
.mt-plot-shell__section-resizer[data-dragging="true"]::before {
  background: rgba(100, 116, 139, 0.36);
}
.mt-plot-shell__section-resizer[hidden] {
  display: none;
}
.mt-plot-shell__section-title {
  color: #526070;
  font-size: 0.72rem;
  font-weight: 700;
  margin-bottom: 0.25rem;
  text-transform: uppercase;
}
.mt-param-row,
.mt-legend-row,
.mt-field-row {
  align-items: center;
  display: flex;
  gap: 0.25rem;
  min-width: 0;
}
.mt-param-row {
  display: flex;
  gap: 0;
  margin: 0;
  width: 100%;
}
.mt-param-row input[type="range"] {
  height: 0.8rem;
  margin: 0;
  max-width: 12rem;
  width: 100%;
}
.mt-param-row input[type="text"],
.mt-field-row input,
.mt-field-row select {
  background: transparent;
  border: 1px solid #d8dee6;
  border-radius: 0.2rem;
  font: inherit;
  min-width: 0;
  padding: 0.12rem 0.25rem;
}
.mt-param-row__value {
  font-size: 0.72rem;
  text-align: right;
  width: 7ch;
}
.mt-param-row__label-value {
  flex: 0 0 auto;
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
  white-space: nowrap;
}
.mt-param-row__slider-stack {
  flex: 0 1 auto;
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
.mt-param-row .mt-text-entry--limit {
  background: transparent;
  border: 0;
  box-shadow: none;
  font-size: 0.68rem;
  height: 0.95rem;
  -webkit-mask-image: linear-gradient(to right, transparent 0, #000 0.55rem);
  mask-image: linear-gradient(to right, transparent 0, #000 0.55rem);
  outline-offset: 0;
  padding: 0;
  width: 5ch;
}
.mt-param-row .mt-text-entry--limit:focus {
  outline: 1px solid rgba(71, 85, 105, 0.45);
}
.mt-param-row .mt-text-entry--minimum {
  padding-left: 0;
  text-align: right;
}
.mt-param-row .mt-text-entry--maximum {
  padding-right: 0;
  text-align: left;
}
.mt-plot-shell[data-layout="compact"] {
  grid-template-columns: 1fr;
}
.mt-plot-shell[data-layout="compact"] .mt-plot-shell__side {
  min-height: 0;
  min-width: 0;
  width: 100%;
}
.mt-plot-shell[data-layout="compact"] .mt-plot-shell__section[data-section="label"],
.mt-plot-shell[data-layout="compact"] .mt-plot-shell__section[data-section="parameter"],
.mt-plot-shell[data-layout="compact"] .mt-plot-shell__section[data-section="info"] {
  flex: 0 1 auto;
  max-height: 12rem;
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
.mt-icon-button {
  align-items: center;
  background: transparent;
  border: 0;
  border-radius: 0.2rem;
  cursor: pointer;
  display: inline-flex;
  height: 1.3rem;
  justify-content: center;
  padding: 0;
  width: 1.3rem;
}
.mt-icon-button svg {
  height: 1rem;
  width: 1rem;
}
.mt-icon-button:hover {
  background: #eef2f7;
}
.mt-legend-row {
  margin: 0;
}
.mt-legend-label {
  flex: 1 1 auto;
  min-width: 0;
  overflow: hidden;
  -webkit-mask-image: linear-gradient(to right, #000 calc(100% - 1.15rem), transparent);
  mask-image: linear-gradient(to right, #000 calc(100% - 1.15rem), transparent);
  white-space: nowrap;
}
.mt-legend-label p {
  overflow: hidden;
  white-space: nowrap;
}
.mt-legend-row[aria-disabled="true"] {
  opacity: 0.55;
}
.mt-legend-marker {
  flex: 0 0 1rem;
  height: 1rem;
  width: 1rem;
}
.mt-markdown p {
  margin: 0;
}
.mt-info-card__segment {
  display: inline;
}
.mt-info-card__segment-view,
.mt-info-card__segment-view > *,
.mt-info-card__segment-view .jupyter-widgets,
.mt-info-card__segment-view .widget-output,
.mt-info-card__segment-view .output,
.mt-info-card__segment-view .output_area,
.mt-info-card__segment-view .output_subarea,
.mt-info-card__segment-view .jp-OutputArea,
.mt-info-card__segment-view .jp-OutputArea-child,
.mt-info-card__segment-view .jp-OutputArea-output,
.mt-info-card__segment-view .jp-RenderedMarkdown,
.mt-info-card__segment-view .jp-RenderedHTMLCommon,
.mt-info-card__segment--markdown p {
  display: inline;
}
.mt-stdout {
  margin: 0;
  overflow: auto;
  white-space: pre-wrap;
}
.mt-status {
  color: inherit;
}
.mt-modal {
  align-items: center;
  background: rgba(15, 23, 42, 0.24);
  display: flex;
  inset: 0;
  justify-content: center;
  padding: 1rem;
  position: absolute;
  z-index: 10;
}
.mt-modal[hidden] {
  display: none;
}
.mt-modal__dialog {
  background: #fff;
  border: 1px solid #c8d0d9;
  border-radius: 0.35rem;
  box-shadow: 0 0.45rem 1.5rem rgba(15, 23, 42, 0.18);
  display: flex;
  flex-direction: column;
  max-height: calc(100% - 2rem);
  max-width: 34rem;
  overflow: hidden;
  width: min(34rem, 100%);
}
.mt-modal__header,
.mt-modal__footer {
  align-items: center;
  display: flex;
  justify-content: space-between;
  padding: 0.75rem;
}
.mt-modal__title {
  color: #0f172a;
  font-size: 0.95rem;
  font-weight: 700;
  line-height: 1.25;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.mt-modal__body {
  display: flex;
  flex-direction: column;
  gap: 0.55rem;
  min-height: 0;
  overflow: auto;
  padding: 0.8rem 0.75rem;
}
.mt-modal__errors {
  background: #fff1f2;
  border: 1px solid #fecdd3;
  border-radius: 0.2rem;
  color: #9f1239;
  line-height: 1.35;
  padding: 0.5rem 0.65rem;
}
.mt-modal__errors ul {
  margin: 0;
  padding-left: 1.1rem;
}
.mt-modal__footer {
  gap: 0.5rem;
  justify-content: flex-end;
}
.mt-modal__button {
  background: #fff;
  border: 1px solid #cbd5e1;
  border-radius: 0.25rem;
  color: #1e293b;
  cursor: pointer;
  font: inherit;
  min-height: 1.8rem;
  padding: 0.24rem 0.65rem;
}
.mt-modal__button:hover {
  background: #f8fafc;
}
.mt-modal__button--primary {
  background: #2563eb;
  border-color: #2563eb;
  color: #fff;
}
.mt-modal__button--primary:hover {
  background: #1d4ed8;
}
.mt-modal__group {
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
  min-width: 0;
}
.mt-modal__fields {
  display: flex;
  flex-direction: column;
  gap: 0.55rem;
  min-width: 0;
}
.mt-field-row-pair {
  display: flex;
  flex-wrap: wrap;
  gap: 0.55rem 1.35rem;
  justify-content: space-between;
  min-width: 0;
}
.mt-field-row-pair .mt-field-row {
  flex: 0 1 auto;
  min-width: min(100%, max-content);
}
.mt-field-row-pair .mt-field-row__label {
  white-space: nowrap;
}
.mt-field-row {
  align-items: center;
  display: flex;
  flex-wrap: nowrap;
  gap: 0.45rem;
  min-height: 1.8rem;
  min-width: 0;
}
.mt-field-row--compact {
  gap: 0.35rem;
}
.mt-field-row__label {
  color: #334155;
  flex: 0 0 auto;
  line-height: 1.8rem;
}
.mt-field-row__control {
  align-items: center;
  display: flex;
  flex: 0 1 auto;
  gap: 0.35rem;
  min-width: 0;
}
.mt-field-row input,
.mt-field-row select,
.mt-dash-select__trigger {
  background: #fff;
  border: 1px solid #cbd5e1;
  border-radius: 0.2rem;
  color: #0f172a;
  font: inherit;
  min-height: 1.8rem;
}
.mt-field-row input:focus,
.mt-field-row select:focus,
.mt-dash-select__trigger:focus-visible {
  outline: 2px solid rgba(37, 99, 235, 0.34);
  outline-offset: 1px;
}
.mt-field-row input[type="text"],
.mt-field-row input[type="number"] {
  min-width: 0;
  padding: 0.16rem 0.35rem;
}
.mt-field-row input[type="text"] {
  width: min(16rem, 100%);
}
.mt-field-row input[type="number"] {
  width: 5.5rem;
}
.mt-field-row input[type="checkbox"] {
  height: 1rem;
  min-height: 1rem;
  width: 1rem;
}
.mt-field-row input[type="color"] {
  flex: 0 0 2.35rem;
  height: 1.8rem;
  padding: 0.1rem;
  width: 2.35rem;
}
.mt-color-field__select {
  flex: 0 0 11ch;
  max-width: 11ch;
}
.mt-opacity-field__slider {
  flex: 0 0 5.5rem;
  min-width: 0;
}
.mt-opacity-field__entry,
.mt-line-width-field__entry {
  flex: 0 0 3.2rem;
  text-align: right;
  width: 3.2rem;
}
.mt-line-width-field__preview {
  align-items: center;
  color: #2563eb;
  display: inline-flex;
  flex: 0 0 2.1rem;
  height: 1.25rem;
  justify-content: center;
}
.mt-line-width-field__preview svg {
  height: 1.25rem;
  width: 2.1rem;
}
.mt-line-width-field__suffix {
  color: #475569;
  flex: 0 0 auto;
  font-size: 0.78rem;
}
.mt-dash-select {
  flex: 0 1 9.5rem;
  min-width: 0;
  position: relative;
  width: 9.5rem;
}
.mt-dash-select__trigger {
  align-items: center;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  padding: 0.18rem 0.4rem;
  width: 100%;
}
.mt-dash-select__selected,
.mt-dash-select__option {
  align-items: center;
  display: flex;
  gap: 0.45rem;
  min-width: 0;
}
.mt-dash-select__preview {
  flex: 0 0 2.75em;
  height: 0.8em;
  width: 2.75em;
}
.mt-dash-select__text {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.mt-dash-select__chevron {
  border-left: 0.3rem solid transparent;
  border-right: 0.3rem solid transparent;
  border-top: 0.4rem solid currentColor;
  flex: 0 0 auto;
  margin-left: 0.5rem;
}
.mt-dash-select__menu {
  background: #fff;
  border: 1px solid #cbd5e1;
  border-radius: 0.2rem;
  box-shadow: 0 0.35rem 0.8rem rgba(15, 23, 42, 0.12);
  list-style: none;
  margin: 0;
  padding: 0.2rem;
  position: fixed;
  z-index: 2147483647;
}
.mt-dash-select__menu[hidden] {
  display: none;
}
.mt-dash-select__option {
  border-radius: 0.2rem;
  cursor: pointer;
  padding: 0.35rem 0.45rem;
}
.mt-dash-select__option[aria-selected="true"] {
  background: #dbeafe;
  color: #1d4ed8;
  font-weight: 600;
}
.mt-dash-select__option:hover,
.mt-dash-select__option:focus {
  background: #eff6ff;
  outline: none;
}
.mt-plot-shell[data-state="disconnected"] {
  box-shadow: inset 0 0 0 0.18rem rgba(100, 116, 139, 0.55);
  opacity: 0.86;
  padding: 0.3rem;
}
@media (max-width: 760px) {
  .mt-plot-shell {
    grid-template-columns: 1fr;
  }
}
"""

    _esm = MARKDOWN_RENDERER_ESM + CHILD_MOUNT_ESM + r"""
function syncOverflowTitle(element, title) {
  if (element.scrollWidth > element.clientWidth) {
    element.title = title;
  } else {
    element.removeAttribute("title");
  }
}

function gearSvg() {
  return `
    <svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Z"></path>
      <path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.6V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1A2 2 0 1 1 4.2 17l.1-.1a1.7 1.7 0 0 0 .3-1.9 1.7 1.7 0 0 0-1.6-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9L4.3 7A2 2 0 1 1 7 4.2l.1.1a1.7 1.7 0 0 0 1.9.3 1.7 1.7 0 0 0 1-1.6V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.6 1.7 1.7 0 0 0 1.9-.3l.1-.1A2 2 0 1 1 19.8 7l-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.6 1h.1a2 2 0 1 1 0 4H21a1.7 1.7 0 0 0-1.6 1Z"></path>
    </svg>`;
}

function speakerSvg(status) {
  if (status === "paused") {
    return `
      <svg viewBox="0 0 24 24" aria-hidden="true" fill="currentColor">
        <rect x="7" y="5" width="3.2" height="14" rx="0.7"></rect>
        <rect x="13.8" y="5" width="3.2" height="14" rx="0.7"></rect>
      </svg>`;
  }
  const waves = status === "playing"
    ? `<path d="M17 9.5c1.1 1.4 1.1 3.6 0 5"></path><path d="M20 7c2.1 2.7 2.1 7.3 0 10"></path>`
    : "";
  return `
    <svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M4 9v6h4l5 4V5L8 9H4Z"></path>
      ${waves}
    </svg>`;
}

function send(model, generationId, type, payload = {}) {
  model.send({ type, generation_id: generationId, ...payload });
}

const SOUND_LONG_PRESS_MS = 650;

function render({ model, el }) {
  const generationId = model.get("generation_id");
  el.classList.add("mt-plot-shell");
  el.classList.add(model.get("root_class") || "mt-host-jupyter");
  el.dataset.host = model.get("host_name") || "jupyter";
  el.dataset.state = model.get("state");

  const main = document.createElement("div");
  main.className = "mt-plot-shell__main";
  const plot = document.createElement("div");
  plot.className = "mt-plot-shell__plot";
  const runtime = document.createElement("div");
  runtime.className = "mt-plot-shell__runtime";
  const side = document.createElement("div");
  side.className = "mt-plot-shell__side";
  const output = document.createElement("div");
  output.className = "mt-plot-shell__output";
  output.style.gridColumn = "1 / -1";
  const modal = document.createElement("div");
  modal.className = "mt-modal";
  modal.hidden = true;

  main.append(plot, runtime);
  el.replaceChildren(main, side, output, modal);

  let plotView = null;
  let runtimeViews = [];
  let controlViews = [];
  let legendLabelViews = new Map();
  let legendRows = new Map();
  let legendLayoutWidth = 0;
  let infoMarkdownViews = new Map();
  let infoCards = new Map();
  let childRenderToken = 0;
  let controlRenderToken = 0;
  let legendRenderToken = 0;
  let infoRenderToken = 0;
  let modalCleanups = [];
  let sectionResizers = [];
  let shellResizeObserver = null;
  let observedShellWidth = 0;

  function syncResponsiveLayout() {
    const width = Math.floor(el.getBoundingClientRect().width);
    if (width <= 0) {
      return;
    }
    el.dataset.layout = width < 760 ? "compact" : "wide";
  }

  function resizePlotlyDescendants() {
    for (const plotlyElement of plot.querySelectorAll(".js-plotly-plot")) {
      if (window.Plotly?.Plots?.resize) {
        window.Plotly.Plots.resize(plotlyElement);
      }
    }
  }

  function schedulePlotResize(view = plotView) {
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        syncResponsiveLayout();
        if (view && typeof view.resize === "function") {
          view.resize();
        }
        resizePlotlyDescendants();
        window.dispatchEvent(new Event("resize"));
      });
    });
  }

  function resizeEntryWidth(entry) {
    return Math.floor(entry?.contentRect?.width ?? 0);
  }

  function schedulePlotResizeForShellWidthChange(entry) {
    const width = resizeEntryWidth(entry);
    if (width <= 0) {
      return;
    }
    if (width === observedShellWidth) {
      return;
    }
    observedShellWidth = width;
    schedulePlotResize();
  }

  function clearModalCleanups() {
    for (const cleanup of modalCleanups.splice(0)) {
      cleanup();
    }
  }

  async function renderChildWidgets() {
    const token = ++childRenderToken;
    const previousPlotView = plotView;
    const previousRuntimeViews = runtimeViews;
    plotView = null;
    runtimeViews = [];
    plot.replaceChildren();
    runtime.replaceChildren();
    disposeView(previousPlotView);
    previousRuntimeViews.forEach(disposeView);

    try {
      const nextPlotModel = model.get("plot_widget");
      if (nextPlotModel) {
        const view = await createWidgetView(model, nextPlotModel, { preferHosted: true });
        if (token !== childRenderToken) {
          disposeView(view);
          return;
        }
        plotView = view;
        if (view.el) {
          view.el.classList.add("mt-plot-shell__plot-widget-view");
          plot.append(view.el);
          schedulePlotResize(view);
        }
      }

      const runtimeWidgetModels = model.get("runtime_widgets") || [];
      for (let index = 0; index < runtimeWidgetModels.length; index += 1) {
        const view = await createWidgetView(model, runtimeWidgetModels[index]);
        if (token !== childRenderToken) {
          disposeView(view);
          return;
        }
        runtimeViews.push(view);
        if (view.el) {
          runtime.append(view.el);
        }
      }
      if (token !== childRenderToken) {
        return;
      }
      schedulePlotResize();
    } catch (error) {
      console.error("[math_toolkit] Failed to render nested plot widget.", error);
      plot.textContent = "Plot widget failed to render.";
    }
  }

  async function renderControlWidgets() {
    const token = ++controlRenderToken;
    const previousControlViews = controlViews;
    controlViews = [];
    controls.body.replaceChildren();
    previousControlViews.forEach(disposeView);

    const widgets = model.get("control_widgets") || [];
    controls.root.hidden = widgets.length === 0;
    syncSectionResizers();
    try {
      for (let index = 0; index < widgets.length; index += 1) {
        const view = await createWidgetView(model, widgets[index]);
        if (token !== controlRenderToken) {
          disposeView(view);
          return;
        }
        controlViews.push(view);
        if (view.el) {
          controls.body.append(view.el);
        }
      }
    } catch (error) {
      console.error("[math_toolkit] Failed to render parameter control widget.", error);
      controls.body.textContent = "Parameter controls failed to render.";
      controls.root.hidden = false;
      syncSectionResizers();
    }
  }

  function section(title, name) {
    const root = document.createElement("section");
    root.className = "mt-plot-shell__section";
    root.dataset.section = name;
    const body = document.createElement("div");
    body.className = "mt-plot-shell__section-body";
    if (title) {
      const heading = document.createElement("div");
      heading.className = "mt-plot-shell__section-title";
      heading.textContent = title;
      root.append(heading);
    }
    root.append(body);
    return { root, body };
  }

  function sectionMinHeight(sectionRoot) {
    const value = Number.parseFloat(getComputedStyle(sectionRoot).minHeight);
    return Number.isFinite(value) ? value : 0;
  }

  function setSectionHeight(sectionRoot, height) {
    const nextHeight = Math.max(sectionMinHeight(sectionRoot), Math.round(height));
    sectionRoot.style.flexBasis = `${nextHeight}px`;
  }

  function makeSectionResizer(before, after) {
    const resizer = document.createElement("div");
    resizer.className = "mt-plot-shell__section-resizer";
    resizer.setAttribute("role", "separator");
    resizer.setAttribute("aria-orientation", "horizontal");
    resizer.title = "Resize sections";

    resizer.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) {
        return;
      }
      event.preventDefault();
      const beforeStart = before.root.getBoundingClientRect().height;
      const afterStart = after.root.getBoundingClientRect().height;
      const beforeMin = sectionMinHeight(before.root);
      const afterMin = sectionMinHeight(after.root);
      const startY = event.clientY;
      resizer.dataset.dragging = "true";
      resizer.setPointerCapture(event.pointerId);

      const drag = (moveEvent) => {
        const delta = moveEvent.clientY - startY;
        const boundedDelta = Math.min(
          Math.max(delta, beforeMin - beforeStart),
          afterStart - afterMin,
        );
        setSectionHeight(before.root, beforeStart + boundedDelta);
        setSectionHeight(after.root, afterStart - boundedDelta);
        schedulePlotResize();
      };

      const stop = (stopEvent) => {
        resizer.removeEventListener("pointermove", drag);
        resizer.removeEventListener("pointerup", stop);
        resizer.removeEventListener("pointercancel", stop);
        if (resizer.hasPointerCapture(stopEvent.pointerId)) {
          resizer.releasePointerCapture(stopEvent.pointerId);
        }
        delete resizer.dataset.dragging;
      };

      resizer.addEventListener("pointermove", drag);
      resizer.addEventListener("pointerup", stop);
      resizer.addEventListener("pointercancel", stop);
    });

    return { root: resizer, before, after };
  }

  function syncSectionResizers() {
    for (const resizer of sectionResizers) {
      resizer.root.hidden = resizer.before.root.hidden || resizer.after.root.hidden;
    }
  }

  const legend = section("", "label");
  const controls = section("", "parameter");
  const info = section("", "info");
  const status = section("", "status");
  const labelParameterResizer = makeSectionResizer(legend, controls);
  const parameterInfoResizer = makeSectionResizer(controls, info);
  sectionResizers = [labelParameterResizer, parameterInfoResizer];
  side.append(
    legend.root,
    labelParameterResizer.root,
    controls.root,
    parameterInfoResizer.root,
    info.root,
    status.root,
  );

  function disabled() {
    return model.get("state") !== "active";
  }

  function markerSvg(marker) {
    const fill = marker.fill_color || "transparent";
    const stroke = marker.border_color || marker.fill_color || "#64748b";
    const dash = marker.border_dash && marker.border_dash !== "solid" ? "3 2" : "";
    return `<svg class="mt-legend-marker" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="5.5" fill="${escapeHtml(fill)}" stroke="${escapeHtml(stroke)}" stroke-width="${Number(marker.border_width || 1.5)}" ${dash ? `stroke-dasharray="${dash}"` : ""}/></svg>`;
  }

  function syncLegendTitles() {
    for (const label of legend.body.querySelectorAll(".mt-legend-label")) {
      syncOverflowTitle(label, label.dataset.tooltip || "");
    }
  }

  async function renderLegend() {
    const token = ++legendRenderToken;
    const items = model.get("legend") || [];
    const labelWidgets = model.get("legend_label_widgets") || [];
    legend.root.hidden = items.length === 0;
    syncSectionResizers();

    let layoutMayHaveChanged = false;
    let spacingMayHaveChanged = false;
    const wantedNodeIds = new Set(items.map((item) => String(item.node_id)));
    for (const [nodeId, row] of Array.from(legendRows.entries())) {
      if (!wantedNodeIds.has(nodeId)) {
        disposeLegendRow(row);
        row.root.remove();
        legendRows.delete(nodeId);
        layoutMayHaveChanged = true;
      }
    }

    const orderedRows = [];
    for (const item of items) {
      if (token !== legendRenderToken) {
        return;
      }
      const nodeId = String(item.node_id);
      let row = legendRows.get(nodeId);
      if (!row) {
        row = createLegendRow();
        legendRows.set(nodeId, row);
        layoutMayHaveChanged = true;
      }
      const rowChange = await syncLegendRow(row, item, labelWidgets, token);
      layoutMayHaveChanged = rowChange === "layout" || layoutMayHaveChanged;
      spacingMayHaveChanged = rowChange === "spacing" || spacingMayHaveChanged;
      orderedRows.push(row.root);
    }

    if (token !== legendRenderToken) {
      return;
    }
    for (let index = 0; index < orderedRows.length; index += 1) {
      const row = orderedRows[index];
      if (legend.body.children[index] !== row) {
        legend.body.insertBefore(row, legend.body.children[index] || null);
        layoutMayHaveChanged = true;
      }
    }
    requestAnimationFrame(() => {
      syncLegendTitles();
      const width = Math.ceil(legend.body.getBoundingClientRect().width);
      if (width > 0) {
        const widthChanged = width !== legendLayoutWidth;
        legendLayoutWidth = width;
        if (layoutMayHaveChanged || (spacingMayHaveChanged && widthChanged)) {
          schedulePlotResize();
        }
      } else if (layoutMayHaveChanged) {
        schedulePlotResize();
      }
    });
  }

  function createLegendRow() {
    const root = document.createElement("div");
    root.className = "mt-legend-row";
    const marker = document.createElement("button");
    marker.type = "button";
    marker.className = "mt-icon-button";
    marker.title = "Toggle plot visibility";
    marker.setAttribute("aria-label", marker.title);
    const sound = document.createElement("button");
    sound.type = "button";
    sound.className = "mt-icon-button";
    const label = document.createElement("div");
    label.className = "mt-markdown mt-legend-label";
    const edit = document.createElement("button");
    edit.type = "button";
    edit.className = "mt-icon-button";
    edit.title = "Edit plot settings";
    edit.setAttribute("aria-label", edit.title);
    edit.innerHTML = gearSvg();
    const row = {
      root,
      marker,
      sound,
      label,
      edit,
      nodeId: null,
      labelMode: "",
      labelMarkdown: "",
      labelViewKey: "",
      labelPayloadKey: "",
      labelWidgetRef: null,
      soundLongPressTimer: null,
      soundLongPressHandled: false,
    };
    marker.addEventListener("click", () => {
      send(model, generationId, "toggle_plot_visibility", { node_id: row.nodeId });
    });
    sound.addEventListener("pointerdown", (event) => {
      if (event.button !== undefined && event.button !== 0) {
        return;
      }
      if (event.pointerId !== undefined && sound.setPointerCapture) {
        sound.setPointerCapture(event.pointerId);
      }
      row.soundLongPressHandled = false;
      window.clearTimeout(row.soundLongPressTimer);
      row.soundLongPressTimer = window.setTimeout(() => {
        row.soundLongPressTimer = null;
        row.soundLongPressHandled = true;
        send(model, generationId, "reset_plot_sound", { node_id: row.nodeId });
      }, SOUND_LONG_PRESS_MS);
    });
    const clearSoundLongPress = () => {
      window.clearTimeout(row.soundLongPressTimer);
      row.soundLongPressTimer = null;
    };
    sound.addEventListener("pointerup", clearSoundLongPress);
    sound.addEventListener("pointercancel", clearSoundLongPress);
    sound.addEventListener("click", (event) => {
      clearSoundLongPress();
      if (row.soundLongPressHandled) {
        event.preventDefault();
        row.soundLongPressHandled = false;
        return;
      }
      send(model, generationId, "toggle_plot_sound", { node_id: row.nodeId });
    });
    sound.addEventListener("contextmenu", (event) => {
      if (row.soundLongPressHandled) {
        event.preventDefault();
      }
    });
    edit.addEventListener("click", () => {
      send(model, generationId, "open_plot_settings", { node_id: row.nodeId });
    });
    root.append(marker, sound, label, edit);
    return row;
  }

  function disposeLegendRow(row) {
    window.clearTimeout(row.soundLongPressTimer);
    row.soundLongPressTimer = null;
    if (row.labelViewKey) {
      disposeView(legendLabelViews.get(row.labelViewKey));
      legendLabelViews.delete(row.labelViewKey);
      row.labelViewKey = "";
    }
  }

  async function syncLegendRow(row, item, labelWidgets, token) {
    let changeKind = "";
    row.nodeId = item.node_id;
    row.root.setAttribute("aria-disabled", item.visible ? "false" : "true");
    row.marker.innerHTML = markerSvg(item.marker || {});
    row.marker.disabled = disabled();

    const hasSound = Boolean(item.sound_playable && item.sound_enabled);
    if (row.sound.hidden === hasSound) {
      changeKind = "layout";
    }
    row.sound.hidden = !hasSound;
    if (hasSound) {
      row.sound.title = item.sound_playing ? "Pause sound" : "Play sound";
      row.sound.setAttribute("aria-label", row.sound.title);
      row.sound.innerHTML = speakerSvg(item.sound_status || (item.sound_playing ? "playing" : "stopped"));
      row.sound.disabled = disabled();
    }

    row.edit.disabled = disabled();
    const labelMarkdown = item.label_markdown || "";
    if (row.labelMarkdown !== labelMarkdown && changeKind !== "layout") {
      changeKind = "spacing";
    }
    row.labelMarkdown = labelMarkdown;
    row.label.dataset.tooltip = labelMarkdown;
    if (Number.isInteger(item.label_widget_index)) {
      const labelWidget = labelWidgets[item.label_widget_index];
      const viewKey = `${item.node_id}:label`;
      if (row.labelMode === "widget" && row.labelViewKey === viewKey && row.labelWidgetRef === labelWidget) {
        return changeKind;
      }
      disposeLegendRow(row);
      row.label.replaceChildren();
      row.labelMode = "widget";
      row.labelWidgetRef = labelWidget;
      row.labelPayloadKey = "";
      if (changeKind !== "layout") {
        changeKind = "spacing";
      }
      try {
        const view = await createWidgetView(model, labelWidget);
        if (token !== legendRenderToken) {
          disposeView(view);
          return changeKind;
        }
        legendLabelViews.set(viewKey, view);
        row.labelViewKey = viewKey;
        if (view?.el) {
          row.label.append(view.el);
        }
      } catch (error) {
        console.error("[math_toolkit] Failed to render legend Markdown label.", error);
        row.labelMode = "payload";
        row.labelWidgetRef = null;
        const payload = item.label_payload || { kind: "markdown", text: labelMarkdown };
        row.labelPayloadKey = JSON.stringify(payload);
        renderMarkdownArea(row.label, payload);
      }
      return changeKind;
    }

    disposeLegendRow(row);
    const payload = item.label_payload || { kind: "markdown", text: labelMarkdown };
    const payloadKey = JSON.stringify(payload);
    if (row.labelMode !== "payload" || row.labelPayloadKey !== payloadKey) {
      row.label.replaceChildren();
      renderMarkdownArea(row.label, payload);
      row.labelMode = "payload";
      row.labelPayloadKey = payloadKey;
      row.labelWidgetRef = null;
      if (changeKind !== "layout") {
        changeKind = "spacing";
      }
    }
    return changeKind;
  }

  async function renderInfo() {
    const token = ++infoRenderToken;
    const items = model.get("info") || [];
    const infoWidgets = model.get("info_markdown_widgets") || [];
    info.root.hidden = items.length === 0;
    syncSectionResizers();

    const wantedCardIds = new Set(items.map((item) => String(item.card_id)));
    for (const [cardId, card] of Array.from(infoCards.entries())) {
      if (!wantedCardIds.has(cardId)) {
        disposeInfoCard(card);
        card.root.remove();
        infoCards.delete(cardId);
      }
    }

    const orderedCards = [];
    for (const item of items) {
      if (token !== infoRenderToken) {
        return;
      }
      const cardId = String(item.card_id);
      let card = infoCards.get(cardId);
      if (!card) {
        card = createInfoCard();
        infoCards.set(cardId, card);
      }
      await syncInfoSlot(card, "title", item, infoWidgets, token);
      await syncInfoSlot(card, "body", item, infoWidgets, token);
      orderedCards.push(card.root);
    }

    if (token !== infoRenderToken) {
      return;
    }
    for (let index = 0; index < orderedCards.length; index += 1) {
      const card = orderedCards[index];
      if (info.body.children[index] !== card) {
        info.body.insertBefore(card, info.body.children[index] || null);
      }
    }
  }

  function createInfoCard() {
    const root = document.createElement("div");
    root.className = "mt-info-card";
    const title = document.createElement("div");
    title.className = "mt-markdown mt-info-card__title";
    const body = document.createElement("div");
    body.className = "mt-markdown mt-info-card__body";
    root.append(title, body);
    return {
      root,
      title: { element: title, mode: "", viewKey: "", payloadKey: "" },
      body: {
        element: body,
        mode: "",
        viewKey: "",
        payloadKey: "",
        segments: new Map(),
      },
    };
  }

  function disposeInfoCard(card) {
    disposeInfoSlot(card.title);
    disposeInfoSlot(card.body);
  }

  function disposeInfoSlot(slot) {
    if (slot.viewKey) {
      disposeView(infoMarkdownViews.get(slot.viewKey));
      infoMarkdownViews.delete(slot.viewKey);
      slot.viewKey = "";
    }
    if (slot.segments) {
      for (const rendered of slot.segments.values()) {
        disposeInfoRenderedSegment(rendered);
      }
      slot.segments.clear();
    }
  }

  async function syncInfoSlot(card, slotName, item, infoWidgets, token) {
    const slot = card[slotName];
    const widgetIndexName = slotName === "title" ? "title_widget_index" : "body_widget_index";
    const payloadName = slotName === "title" ? "title_markdown_payload" : "markdown_payload";
    const fallbackText = slotName === "title"
      ? (item.title_markdown ? `**${item.title_markdown}**` : "")
      : (item.markdown || "");
    const fallbackPayload = item[payloadName] || { kind: "markdown", text: fallbackText };
    const visible = slotName !== "title" || Boolean(item.title_markdown);
    slot.element.hidden = !visible;
    if (!visible) {
      disposeInfoSlot(slot);
      slot.element.replaceChildren();
      slot.mode = "";
      slot.payloadKey = "";
      slot.widgetRef = null;
      return;
    }

    if (slotName === "body" && Array.isArray(item.segments) && item.segments.length) {
      await syncInfoSegments(slot, item.segments, infoWidgets, token, item.card_id);
      return;
    }

    if (Number.isInteger(item[widgetIndexName])) {
      const widgetRef = infoWidgets[item[widgetIndexName]];
      const viewKey = `${item.card_id}:${slotName}`;
      if (slot.mode === "widget" && slot.viewKey === viewKey && slot.widgetRef === widgetRef) {
        return;
      }
      disposeInfoSlot(slot);
      slot.element.replaceChildren();
      slot.mode = "widget";
      slot.widgetRef = widgetRef;
      slot.payloadKey = "";
      await mountInfoMarkdownWidget(slot, viewKey, widgetRef, token, fallbackPayload);
      return;
    }

    disposeInfoSlot(slot);
    const payloadKey = JSON.stringify(fallbackPayload);
    if (slot.mode !== "payload" || slot.payloadKey !== payloadKey) {
      slot.element.replaceChildren();
      renderMarkdownArea(slot.element, fallbackPayload);
      slot.mode = "payload";
      slot.payloadKey = payloadKey;
      slot.widgetRef = null;
    }
  }

  async function syncInfoSegments(slot, segments, infoWidgets, token, cardId) {
    if (slot.mode !== "segments") {
      disposeInfoSlot(slot);
      slot.element.replaceChildren();
      slot.mode = "segments";
      slot.payloadKey = "";
      slot.widgetRef = null;
    }

    const wantedKeys = new Set(segments.map((segment) => String(segment.index)));
    for (const [key, rendered] of Array.from(slot.segments.entries())) {
      if (!wantedKeys.has(key)) {
        disposeInfoRenderedSegment(rendered);
        rendered.element.remove();
        slot.segments.delete(key);
      }
    }

    const orderedElements = [];
    for (const segment of segments) {
      const key = String(segment.index);
      let rendered = slot.segments.get(key);
      if (!rendered) {
        rendered = createInfoSegment();
        slot.segments.set(key, rendered);
      }
      await syncInfoSegment(rendered, segment, infoWidgets, token, cardId);
      orderedElements.push(rendered.element);
    }

    for (let index = 0; index < orderedElements.length; index += 1) {
      const element = orderedElements[index];
      if (slot.element.children[index] !== element) {
        slot.element.insertBefore(element, slot.element.children[index] || null);
      }
    }
  }

  function createInfoSegment() {
    const element = document.createElement("span");
    element.className = "mt-info-card__segment";
    return {
      element,
      mode: "",
      payloadKey: "",
      text: "",
      viewKey: "",
      widgetRef: null,
    };
  }

  function disposeInfoRenderedSegment(rendered) {
    if (rendered.viewKey) {
      disposeView(infoMarkdownViews.get(rendered.viewKey));
      infoMarkdownViews.delete(rendered.viewKey);
      rendered.viewKey = "";
    }
    rendered.widgetRef = null;
  }

  async function syncInfoSegment(rendered, segment, infoWidgets, token, cardId) {
    const payload = segment.markdown_payload || { kind: "markdown", text: segment.text || "" };
    if (Number.isInteger(segment.widget_index)) {
      const widgetRef = infoWidgets[segment.widget_index];
      const viewKey = `${cardId}:segment:${segment.index}`;
      if (
        rendered.mode === "widget"
        && rendered.viewKey === viewKey
        && rendered.widgetRef === widgetRef
      ) {
        return;
      }
      disposeInfoRenderedSegment(rendered);
      rendered.element.className = "mt-info-card__segment mt-info-card__segment--markdown";
      rendered.element.replaceChildren();
      rendered.mode = "widget";
      rendered.widgetRef = widgetRef;
      rendered.payloadKey = "";
      rendered.text = "";
      await mountInfoMarkdownWidget(rendered, viewKey, widgetRef, token, payload);
      return;
    }

    const payloadKey = JSON.stringify(payload);
    if (rendered.mode !== "payload" || rendered.payloadKey !== payloadKey) {
      disposeInfoRenderedSegment(rendered);
      rendered.element.className = "mt-info-card__segment mt-info-card__segment--markdown";
      rendered.element.replaceChildren();
      renderMarkdownArea(rendered.element, payload);
      rendered.mode = "payload";
      rendered.payloadKey = payloadKey;
      rendered.text = "";
    }
  }

  async function mountInfoMarkdownWidget(slot, viewKey, widgetRef, token, fallbackPayload) {
    if (!widgetRef) {
      renderMarkdownArea(slot.element, fallbackPayload);
      slot.mode = "payload";
      slot.payloadKey = JSON.stringify(fallbackPayload);
      slot.widgetRef = null;
      return;
    }
    try {
      const view = await createWidgetView(model, widgetRef);
      if (token !== infoRenderToken) {
        disposeView(view);
        return;
      }
      infoMarkdownViews.set(viewKey, view);
      slot.viewKey = viewKey;
      if (view?.el) {
        view.el.classList?.add("mt-info-card__segment-view");
        slot.element.append(view.el);
      }
    } catch (error) {
      console.error("[math_toolkit] Failed to render info Markdown.", error);
      slot.mode = "payload";
      slot.payloadKey = JSON.stringify(fallbackPayload);
      slot.widgetRef = null;
      renderMarkdownArea(slot.element, fallbackPayload);
    }
  }

  function renderStatus() {
    const current = model.get("status") || {};
    status.root.hidden = !current.message;
    syncSectionResizers();
    status.body.replaceChildren();
    if (current.message) {
      const div = document.createElement("div");
      div.className = "mt-status";
      div.textContent = current.message;
      status.body.append(div);
    }
  }

  function renderOutput() {
    const items = model.get("output_items") || [];
    output.hidden = items.length === 0;
    output.replaceChildren(...items.map(blockElement));
  }

  function sendModalField(field, value) {
    send(model, generationId, "modal_field_changed", { field_id: field.id, value });
  }

  function optionFor(field, value) {
    const options = field.options || [];
    return options.find((option) => option.value === value) || options[0] || {
      label: String(value ?? ""),
      value: String(value ?? ""),
      dasharray: "",
    };
  }

  function dashSvg(option) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "mt-dash-select__preview");
    svg.setAttribute("viewBox", "0 0 48 16");
    svg.setAttribute("aria-hidden", "true");
    svg.setAttribute("focusable", "false");

    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", "3");
    line.setAttribute("y1", "8");
    line.setAttribute("x2", "45");
    line.setAttribute("y2", "8");
    line.setAttribute("stroke", "currentColor");
    line.setAttribute("stroke-width", "2");
    line.setAttribute("stroke-linecap", "round");
    if (option.dasharray) {
      line.setAttribute("stroke-dasharray", option.dasharray);
    }
    svg.append(line);
    return svg;
  }

  function dashValueContent(option) {
    const wrapper = document.createElement("span");
    wrapper.className = "mt-dash-select__selected";
    wrapper.append(dashSvg(option));

    const text = document.createElement("span");
    text.className = "mt-dash-select__text";
    text.textContent = option.label;
    wrapper.append(text);
    return wrapper;
  }

  function numericFieldControl(field, options = {}) {
    const input = document.createElement("input");
    input.type = field.kind === "text" ? "text" : "number";
    input.value = field.value ?? "";
    if (options.min !== undefined) {
      input.min = String(options.min);
    }
    if (options.max !== undefined) {
      input.max = String(options.max);
    }
    if (options.step !== undefined) {
      input.step = String(options.step);
    }
    if (options.className) {
      input.className = options.className;
    }
    input.addEventListener("change", () => sendModalField(field, input.value));
    return input;
  }

  function checkboxFieldControl(field) {
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = Boolean(field.value);
    input.addEventListener("change", () => sendModalField(field, input.checked));
    return input;
  }

  function readonlyFieldControl(field) {
    const input = document.createElement("input");
    input.type = "text";
    input.value = field.value ?? "";
    input.readOnly = true;
    input.disabled = true;
    return input;
  }

  function colorFieldControl(field) {
    const wrapper = document.createElement("div");
    wrapper.className = "mt-field-row__control mt-color-field";
    const picker = document.createElement("input");
    picker.type = "color";
    picker.value = field.meta?.picker || "#1f77b4";
    const select = document.createElement("select");
    select.className = "mt-color-field__select";

    for (const option of field.options || []) {
      const item = document.createElement("option");
      item.value = option.value;
      item.textContent = option.label;
      item.dataset.hex = option.hex || "";
      select.append(item);
    }
    select.value = field.value;

    function syncPickerDraft() {
      if (!Array.from(select.options).some((option) => option.value === picker.value)) {
        const custom = document.createElement("option");
        custom.value = picker.value;
        custom.textContent = picker.value;
        custom.dataset.hex = picker.value;
        select.prepend(custom);
      }
      select.value = picker.value;
    }
    picker.addEventListener("input", syncPickerDraft);
    picker.addEventListener("change", () => {
      syncPickerDraft();
      sendModalField(field, picker.value);
    });
    select.addEventListener("change", () => {
      const selected = select.selectedOptions[0];
      if (selected?.dataset.hex) {
        picker.value = selected.dataset.hex;
      }
      sendModalField(field, select.value);
    });

    wrapper.append(picker, select);
    return wrapper;
  }

  function opacityFieldControl(field) {
    const wrapper = document.createElement("div");
    wrapper.className = "mt-field-row__control mt-opacity-field";
    const slider = document.createElement("input");
    slider.type = "range";
    slider.className = "mt-opacity-field__slider";
    slider.min = "0";
    slider.max = "1";
    slider.step = "0.1";
    slider.value = field.value ?? "1";
    const entry = document.createElement("input");
    entry.type = "text";
    entry.className = "mt-opacity-field__entry";
    entry.value = String(field.value ?? "1");

    // Keep active slider motion browser-local. Sending every input event to
    // Python republishes the modal payload and replaces the dragged node.
    slider.addEventListener("input", () => {
      entry.value = Number(slider.value).toFixed(1);
    });
    slider.addEventListener("change", () => {
      entry.value = Number(slider.value).toFixed(1);
      sendModalField(field, entry.value);
    });
    entry.addEventListener("change", () => {
      const value = Number(entry.value);
      if (Number.isFinite(value) && value >= 0 && value <= 1) {
        slider.value = String(value);
        entry.value = value.toFixed(1);
      }
      sendModalField(field, entry.value);
    });

    wrapper.append(slider, entry);
    return wrapper;
  }

  function widthFieldControl(field) {
    const wrapper = document.createElement("div");
    wrapper.className = "mt-field-row__control mt-line-width-field";
    const preview = document.createElement("span");
    preview.className = "mt-line-width-field__preview";
    const entry = document.createElement("input");
    entry.type = "number";
    entry.className = "mt-line-width-field__entry";
    entry.min = "0";
    entry.step = "0.5";
    entry.value = field.value ?? "2";
    const suffix = document.createElement("span");
    suffix.className = "mt-line-width-field__suffix";
    suffix.textContent = "px";

    function syncPreview() {
      const value = Number(entry.value);
      const width = Math.min(Math.max(Number.isFinite(value) ? value : 0.5, 0.5), 12);
      preview.innerHTML = `
        <svg viewBox="0 0 32 20" aria-label="Line width preview" role="img" focusable="false">
          <line x1="2" y1="10" x2="30" y2="10" stroke="currentColor" stroke-width="${width}" stroke-linecap="round"></line>
        </svg>`;
    }

    entry.addEventListener("input", syncPreview);
    entry.addEventListener("change", () => sendModalField(field, entry.value));
    syncPreview();
    wrapper.append(preview, entry, suffix);
    return wrapper;
  }

  function dashFieldControl(field) {
    const root = document.createElement("div");
    root.className = "mt-dash-select";
    const trigger = document.createElement("button");
    trigger.type = "button";
    trigger.className = "mt-dash-select__trigger";
    trigger.setAttribute("aria-haspopup", "listbox");
    trigger.setAttribute("aria-expanded", "false");
    const selectedSlot = document.createElement("span");
    const chevron = document.createElement("span");
    chevron.className = "mt-dash-select__chevron";
    chevron.setAttribute("aria-hidden", "true");
    trigger.append(selectedSlot, chevron);

    const menu = document.createElement("ul");
    menu.className = "mt-dash-select__menu";
    menu.setAttribute("role", "listbox");
    menu.hidden = true;

    function closeMenu() {
      menu.hidden = true;
      trigger.setAttribute("aria-expanded", "false");
    }

    function positionMenu() {
      const rect = trigger.getBoundingClientRect();
      menu.style.left = `${rect.left}px`;
      menu.style.top = `${rect.bottom + 3}px`;
      menu.style.minWidth = `${rect.width}px`;
    }

    function openMenu() {
      positionMenu();
      menu.hidden = false;
      trigger.setAttribute("aria-expanded", "true");
    }

    function setSelected(value) {
      const selected = optionFor(field, value);
      selectedSlot.replaceChildren(dashValueContent(selected));
      for (const item of menu.querySelectorAll("[role='option']")) {
        item.setAttribute(
          "aria-selected",
          item.dataset.value === selected.value ? "true" : "false"
        );
      }
    }

    for (const option of field.options || []) {
      const item = document.createElement("li");
      item.className = "mt-dash-select__option";
      item.dataset.value = option.value;
      item.setAttribute("role", "option");
      item.tabIndex = -1;
      item.append(dashSvg(option));
      const text = document.createElement("span");
      text.className = "mt-dash-select__text";
      text.textContent = option.label;
      item.append(text);
      item.addEventListener("click", () => {
        setSelected(option.value);
        closeMenu();
        sendModalField(field, option.value);
      });
      menu.append(item);
    }

    trigger.addEventListener("click", () => {
      if (menu.hidden) {
        openMenu();
        return;
      }
      closeMenu();
    });
    trigger.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        closeMenu();
      }
    });
    const outsideClick = (event) => {
      if (!root.contains(event.target) && !menu.contains(event.target)) {
        closeMenu();
      }
    };
    const reposition = () => {
      if (!menu.hidden) {
        positionMenu();
      }
    };
    document.addEventListener("click", outsideClick);
    window.addEventListener("resize", reposition);
    window.addEventListener("scroll", reposition, true);
    setSelected(field.value);
    root.append(trigger);
    document.body.append(menu);
    modalCleanups.push(() => {
      document.removeEventListener("click", outsideClick);
      window.removeEventListener("resize", reposition);
      window.removeEventListener("scroll", reposition, true);
      menu.remove();
    });
    return root;
  }

  function fieldControl(field) {
    if (field.kind === "readonly") {
      return readonlyFieldControl(field);
    }
    if (field.kind === "checkbox") {
      return checkboxFieldControl(field);
    }
    if (field.kind === "color") {
      return colorFieldControl(field);
    }
    if (field.kind === "opacity") {
      return opacityFieldControl(field);
    }
    if (field.kind === "line_width") {
      return widthFieldControl(field);
    }
    if (field.kind === "dash") {
      return dashFieldControl(field);
    }
    if (field.kind === "choice") {
      const input = document.createElement("select");
      for (const option of field.options || []) {
        const item = document.createElement("option");
        item.value = option.value;
        item.textContent = option.label;
        input.append(item);
      }
      input.value = field.value;
      input.addEventListener("change", () => sendModalField(field, input.value));
      return input;
    }
    if (field.kind === "sample_count") {
      return numericFieldControl(field, { min: 2, step: 1 });
    }
    if (field.kind === "positive_float") {
      return numericFieldControl(field, { min: 0, step: "any" });
    }
    return numericFieldControl(field, { step: field.kind === "float" ? "any" : undefined });
  }

  function renderModal() {
    const current = model.get("modal") || { state: "closed" };
    clearModalCleanups();
    modal.hidden = current.state !== "open";
    if (modal.hidden) {
      modal.replaceChildren();
      return;
    }
    const dialog = document.createElement("div");
    dialog.className = "mt-modal__dialog";
    const header = document.createElement("div");
    header.className = "mt-modal__header";
    const title = document.createElement("strong");
    title.className = "mt-modal__title";
    title.textContent = current.title || "Configuration";
    const close = document.createElement("button");
    close.type = "button";
    close.className = "mt-modal__button";
    close.textContent = "Close";
    close.addEventListener("click", () => send(model, generationId, "modal_close"));
    header.append(title, close);
    const body = document.createElement("div");
    body.className = "mt-modal__body";
    if (current.errors && current.errors.length) {
      const errors = document.createElement("div");
      errors.className = "mt-modal__errors";
      const list = document.createElement("ul");
      for (const message of current.errors) {
        const item = document.createElement("li");
        item.textContent = message;
        list.append(item);
      }
      errors.append(list);
      body.prepend(errors);
    }
    const fields = document.createElement("div");
    fields.className = "mt-modal__fields";
    const modalFields = current.fields || [];

    function fieldRow(field) {
      const row = document.createElement("label");
      row.className = field.kind === "checkbox" ? "mt-field-row mt-field-row--compact" : "mt-field-row";
      row.classList.add(`mt-field-row--${field.kind.replaceAll("_", "-")}`);
      const label = document.createElement("span");
      label.className = "mt-field-row__label";
      label.textContent = field.label;
      const control = fieldControl(field);
      if (!control.classList.contains("mt-field-row__control")) {
        const controlWrapper = document.createElement("span");
        controlWrapper.className = "mt-field-row__control";
        controlWrapper.append(control);
        row.append(label, controlWrapper);
      } else {
        row.append(label, control);
      }
      return row;
    }

    for (let index = 0; index < modalFields.length; index += 1) {
      const field = modalFields[index];
      const nextField = modalFields[index + 1];
      const paired =
        (field.kind === "color" && nextField?.kind === "line_width") ||
        (field.kind === "opacity" && nextField?.kind === "dash");
      if (paired) {
        const pair = document.createElement("div");
        pair.className = "mt-field-row-pair";
        pair.append(fieldRow(field), fieldRow(nextField));
        fields.append(pair);
        index += 1;
        continue;
      }
      fields.append(fieldRow(field));
    }
    body.append(fields);
    const footer = document.createElement("div");
    footer.className = "mt-modal__footer";
    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.className = "mt-modal__button";
    cancel.textContent = "Cancel";
    cancel.addEventListener("click", () => send(model, generationId, "modal_cancel"));
    const apply = document.createElement("button");
    apply.type = "button";
    apply.className = "mt-modal__button mt-modal__button--primary";
    apply.textContent = "Apply";
    apply.addEventListener("click", () => send(model, generationId, "modal_apply"));
    footer.append(cancel, apply);
    dialog.append(header, body, footer);
    modal.replaceChildren(dialog);
  }

  function renderState() {
    el.dataset.state = model.get("state");
  }

  function syncState() {
    renderState();
    renderLegend();
  }

  function syncAll() {
    renderState();
    renderLegend();
    renderInfo();
    renderStatus();
    renderOutput();
    renderModal();
  }

  model.on("change:state", syncState);
  model.on("change:legend", renderLegend);
  model.on("change:legend_label_widgets", renderLegend);
  model.on("change:info", renderInfo);
  model.on("change:info_markdown_widgets", renderInfo);
  model.on("change:status", renderStatus);
  model.on("change:output_items", renderOutput);
  model.on("change:modal", renderModal);
  model.on("change:plot_widget", renderChildWidgets);
  model.on("change:runtime_widgets", renderChildWidgets);
  model.on("change:control_widgets", renderControlWidgets);
  window.addEventListener("resize", syncLegendTitles);
  if (typeof ResizeObserver === "function") {
    shellResizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        schedulePlotResizeForShellWidthChange(entry);
      }
    });
    shellResizeObserver.observe(el);
  }
  syncResponsiveLayout();
  syncAll();
  renderChildWidgets();
  renderControlWidgets();

  return () => {
    childRenderToken += 1;
    controlRenderToken += 1;
    infoRenderToken += 1;
    clearModalCleanups();
    model.off("change:state", syncState);
    model.off("change:legend", renderLegend);
    model.off("change:legend_label_widgets", renderLegend);
    model.off("change:info", renderInfo);
    model.off("change:info_markdown_widgets", renderInfo);
    model.off("change:status", renderStatus);
    model.off("change:output_items", renderOutput);
    model.off("change:modal", renderModal);
    model.off("change:plot_widget", renderChildWidgets);
    model.off("change:runtime_widgets", renderChildWidgets);
    model.off("change:control_widgets", renderControlWidgets);
    shellResizeObserver?.disconnect();
    shellResizeObserver = null;
    window.removeEventListener("resize", syncLegendTitles);
    disposeView(plotView);
    runtimeViews.forEach(disposeView);
    controlViews.forEach(disposeView);
    legendLabelViews.forEach(disposeView);
    infoMarkdownViews.forEach(disposeView);
    plotView = null;
    runtimeViews = [];
    controlViews = [];
    legendLabelViews = new Map();
    legendRows = new Map();
    legendLayoutWidth = 0;
    infoMarkdownViews = new Map();
    infoCards = new Map();
  };
}

export default { render };
"""

    def __init__(
        self,
        *,
        generation_id: int,
        plot_widget: object | None = None,
        runtime_widgets: tuple[object, ...] = (),
        host_name: str = "jupyter",
        root_class: str = "mt-host-jupyter",
        child_mount_policy: dict[str, object] | None = None,
        markdown_policy: dict[str, object] | None = None,
    ) -> None:
        """Create a shell for a display generation."""

        super().__init__()
        self.generation_id = generation_id
        self.plot_widget = plot_widget
        self.runtime_widgets = list(runtime_widgets)
        self.host_name = host_name
        self.root_class = root_class
        self.child_mount_policy = (
            {} if child_mount_policy is None else dict(child_mount_policy)
        )
        self.markdown_policy = {} if markdown_policy is None else dict(markdown_policy)
        add_class = getattr(self, "add_class", None)
        if callable(add_class):
            add_class("mt-plot-shell")
            add_class(root_class)

    def set_runtime_widgets(self, widgets: tuple[object, ...]) -> None:
        """Publish hidden helper widgets rendered inside the shell root."""

        if tuple(self.runtime_widgets) == widgets:
            return
        self.runtime_widgets = list(widgets)

    def set_controls(self, controls: tuple[Widget, ...]) -> None:
        """Publish ordered parameter control child widgets."""

        if tuple(self.control_widgets) == controls:
            return
        self.control_widgets = list(controls)

    def set_legend(
        self,
        legend: tuple[dict[str, object], ...],
        label_widgets: tuple[Widget, ...] = (),
    ) -> None:
        """Publish ordered legend row payloads."""

        self.legend = list(legend)
        self.legend_label_widgets = list(label_widgets)

    def set_info(
        self,
        info: tuple[dict[str, object], ...],
        markdown_widgets: tuple[Widget, ...] = (),
    ) -> None:
        """Publish ordered info card payloads."""

        next_info = list(info)
        if self.info != next_info:
            self.info = next_info
        if tuple(self.info_markdown_widgets) != markdown_widgets:
            self.info_markdown_widgets = list(markdown_widgets)

    def set_status(self, message: str | None, *, kind: str = "info") -> None:
        """Publish or clear a status message."""

        self.status = {} if not message else {"message": str(message), "kind": kind}

    def set_output(self, items: tuple[dict[str, str], ...]) -> None:
        """Publish ordered message output blocks."""

        self.output_items = list(items)

    def set_modal(self, payload: dict[str, object]) -> None:
        """Publish modal state."""

        self.modal = dict(payload)

    def set_disconnected(self, message: str) -> None:
        """Mark the shell as disconnected and show the reason."""

        self.state = "disconnected"
        self.set_status(message, kind="warning")

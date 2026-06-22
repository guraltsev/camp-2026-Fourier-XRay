"""Share anywidget child mounting helpers across toolkit frontend widgets."""

from __future__ import annotations

__all__ = ["CHILD_MOUNT_ESM"]


# The policy is supplied by the Python host adapter through synced traits. This
# keeps Marimo hosted-node lookup and Jupyter widget-manager creation in one
# helper instead of copying host heuristics into every composed widget.
CHILD_MOUNT_ESM = r"""
function childMountPolicy(model) {
  return {
    hosted_node_lookup: false,
    hosted_node_timeout_ms: 0,
    widget_manager_view: true,
    local_anywidget_view: true,
    ...(model.get?.("child_mount_policy") || {}),
  };
}

function disposeView(view) {
  if (!view) {
    return;
  }
  if (typeof view.remove === "function") {
    view.remove();
  } else if (view.el && typeof view.el.remove === "function") {
    view.el.remove();
  }
}

async function resolveWidgetModel(model, widgetRef) {
  if (!widgetRef) {
    return null;
  }
  if (typeof widgetRef === "string") {
    const manager = model.widget_manager;
    if (!manager) {
      throw new Error("The active widget manager cannot resolve nested widget models.");
    }
    const modelId = widgetRef.startsWith("IPY_MODEL_")
      ? widgetRef.slice("IPY_MODEL_".length)
      : widgetRef;
    if (typeof manager.get_model !== "function") {
      throw new Error("The active widget manager cannot resolve nested widget models.");
    }
    return await manager.get_model(modelId);
  }
  return widgetRef;
}

function modelIdForWidgetRef(widgetRef) {
  if (typeof widgetRef === "string") {
    return widgetRef.startsWith("IPY_MODEL_")
      ? widgetRef.slice("IPY_MODEL_".length)
      : widgetRef;
  }
  return widgetRef?.model_id || widgetRef?.modelId || widgetRef?.id || null;
}

function hostedMarimoWidgetView(modelId) {
  if (!modelId) {
    return null;
  }
  for (const candidate of document.querySelectorAll("marimo-anywidget")) {
    const rawModelId = candidate.getAttribute("data-model-id");
    let candidateModelId = rawModelId;
    try {
      candidateModelId = JSON.parse(rawModelId);
    } catch {
      candidateModelId = rawModelId;
    }
    if (candidateModelId === modelId) {
      return {
        el: candidate,
        remove() {},
      };
    }
  }
  return null;
}

async function waitForHostedMarimoWidgetView(modelId, timeoutMs) {
  const started = performance.now();
  while (performance.now() - started < timeoutMs) {
    const view = hostedMarimoWidgetView(modelId);
    if (view) {
      return view;
    }
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  return null;
}

function widgetManagerCandidates(model, widgetRef, widgetModel) {
  return [
    model.widget_manager,
    widgetModel?.widget_manager,
    widgetRef?.widget_manager,
    widgetModel?.model?.widget_manager,
    widgetRef?.model?.widget_manager,
  ].filter(Boolean);
}

async function createWidgetView(model, widgetRef, options = {}) {
  const policy = childMountPolicy(model);
  const modelId = modelIdForWidgetRef(widgetRef);
  if (policy.hosted_node_lookup) {
    const hostedView = options.preferHosted
      ? await waitForHostedMarimoWidgetView(modelId, policy.hosted_node_timeout_ms || 3000)
      : hostedMarimoWidgetView(modelId);
    if (hostedView) {
      return hostedView;
    }
  }

  const widgetModel = await resolveWidgetModel(model, widgetRef);
  if (!widgetModel) {
    return null;
  }
  let firstError = null;
  if (policy.widget_manager_view) {
    for (const manager of widgetManagerCandidates(model, widgetRef, widgetModel)) {
      if (typeof manager.create_view === "function") {
        try {
          return await manager.create_view(widgetModel);
        } catch (error) {
          firstError ??= error;
        }
      }
      if (typeof manager.createView === "function") {
        try {
          return await manager.createView(widgetModel);
        } catch (error) {
          firstError ??= error;
        }
      }
    }
  }
  if (firstError && !policy.local_anywidget_view) {
    throw firstError;
  }
  if (!policy.local_anywidget_view) {
    throw new Error("The active widget manager cannot create nested widget views.");
  }
  return await createAnywidgetView(widgetModel);
}

async function createAnywidgetView(widgetModel) {
  const source = widgetModel.get?.("_esm");
  if (!source) {
    throw new Error("The active widget manager cannot create nested widget views.");
  }

  const container = document.createElement("div");
  const mount = document.createElement("div");
  container.style.display = "contents";
  const style = document.createElement("style");
  const css = widgetModel.get?.("_css");
  if (css) {
    style.textContent = css;
    container.append(style);
  }
  container.append(mount);

  const url = source.startsWith("data:") || source.startsWith("http") || source.startsWith("/@file/")
    ? source
    : URL.createObjectURL(new Blob([source], { type: "text/javascript" }));
  let cleanup = null;
  let initializeCleanup = null;
  try {
    const module = await import(url);
    const widget = module.default;
    const implementation = typeof widget === "function" ? await widget() : widget;
    initializeCleanup = await implementation?.initialize?.({
      model: widgetModel,
      experimental: {},
    });
    cleanup = await implementation?.render?.({
      model: widgetModel,
      el: mount,
      experimental: {},
    });
  } finally {
    if (url.startsWith("blob:")) {
      URL.revokeObjectURL(url);
    }
  }

  return {
    el: container,
    remove() {
      if (typeof cleanup === "function") {
        cleanup();
      }
      if (typeof initializeCleanup === "function") {
        initializeCleanup();
      }
      container.remove();
    },
  };
}
"""

# Interactive plotting

`math_toolkit.plotting` adds notebook-first commands for sampled two-dimensional
curves, scalar fields, filled domains, and simple parametric curves.

```python
from math_toolkit import contour_plot, domain_plot, parametric_plot, plot

plot(sin(a * x), x, name="trig")
plot(cos(a * x), name="trig")

contour_plot(x**2 + y**2, x, y, name="levels")
domain_plot(1 - x**2 - y**2, x, y)
parametric_plot((cos(t), sin(t)), (t, 0, 2*pi))
```

Named calls update the plot with the same name inside the same figure when the
kind is unchanged. Changing kind for a name removes the previous plot and
creates a fresh one, so curve style does not carry into a contour, heatmap,
domain, or parametric plot.

## Explicit variables

Plot variables are never inferred from expression free symbols. Declare them
in every new plot call:

```python
plot(sin(x), x)
plot(sin(x), (x, -10, 10))

temperature_plot(sin(x) * cos(y), x, y)
contour_plot(x**2 + y**2, (x, -2, 2), (y, -2, 2))
domain_plot(x**2 + y**2 < 1, x, y)
parametric_plot((cos(t), sin(t)), (t, 0, 2*pi))
```

Named updates may omit domains after the plot exists:

```python
contour_plot(x**2 + y**2, x, y, name="levels")
contour_plot(x**2 - y**2, name="levels")
```

## View-aware plots

`plot`, `temperature_plot`, `contour_plot`, and `domain_plot` are view-aware.
If an axis is declared as just a symbol, the initial sampled range comes from
the default visible view. If finite bounds are declared, the active view is
clipped to those bounds.

```python
plot(sin(x), x)
temperature_plot(sin(x) * cos(y), x, y)
temperature_plot(sin(x) * cos(y), (x, -2, 2), (y, -1, 3))
```

Panning or zooming a Plotly x-axis resamples curves and two-dimensional plots.
Panning or zooming a y-axis resamples two-dimensional plots. Empty
view/domain intersections render empty sampled data.

The visible axes and the home/reset axes are Python-owned figure state, not
Plotly autorange guesses. Plotly receives explicit `xaxis.range` and
`yaxis.range` values, so the Plotly home button returns to the same home view
that Python uses for sampling. `fig.view.reset()` performs the same reset from
Python:

```python
fig = figure("view-demo")
fig.plot(sin(x), x)
fig.view.range = {"x": (-2, 2), "y": (-1.5, 1.5)}
fig.view.home_range = fig.view.range
fig.view.reset()
```

Views are figure-owned handles with stable ids and optional names. A figure has
one current view, and unnamed view commands target it:

```python
overview = fig.view("overview")
fig.view.current = overview
fig.view.range = {"x": (-10, 10), "y": (-3, 3)}
fig.view.home_x_range = (-12, 12)
fig.view.reset()
overview.id
```

`fig.view.range = ...` changes the visible axes of the current view while
preserving that view's home ranges, so the Plotly home button and
`fig.view.reset()` can still return to the previous home view. Assign both
axes with `{"x": (xmin, xmax), "y": (ymin, ymax)}`, or assign one axis with
`fig.view.x_range`, `fig.view.y_range`, `fig.view.home_x_range`, and
`fig.view.home_y_range`. Range mutation is intentionally property-based under
`fig.view`; lower-level setter methods are private implementation hooks.

## Curve audio

Ordinary scalar curves returned by `plot(...)` expose a `sound` namespace.
Other plot types do not expose playback. Playback is figure-owned, so starting
one curve stops any other active sound in the same figure:

```python
fig = figure("sound")
tone = plot(sin(220 * pi * x), x, name="tone", figure=fig)

tone.sound.play(sample_rate=48_000, chunk_frames=2_048)
tone.sound.pause()
tone.sound.resume()
tone.sound.time = 0
tone.sound.stop()
```

New curve plots show a loudspeaker button in the figure legend by default.
Clicking the button starts or resumes the curve sound; clicking it while
playing pauses playback. Set `tone.sound.enabled = False` to hide that
curve's legend sound control.

Figures expose selection and transport commands without a figure-level
`play(...)` method:

```python
fig.sound.current = "tone"
fig.sound.resume()
fig.sound.time
fig.sound.stop()
fig.sound.state()
```

Audio samples are streamed through a hidden anywidget/Web Audio adapter in
small PCM chunks from the curve's mathematical source state. The browser
emitter prefers an `AudioWorkletProcessor` so queued PCM is consumed on the
audio thread; frontends that cannot load an AudioWorklet fall back to scheduled
`AudioBufferSourceNode` chunks. The audio sampler does not reuse Plotly traces,
visual sample buffers, labels, styles, legends, or visible pan/zoom ranges.
Parameter and expression changes affect subsequent chunks and use phase
matching during continuous playback. Curve sound normalization is enabled by
default and can be toggled with `curve.sound.normalization = False` or
`curve.sound.play(normalization=False)`.

## Fields

`temperature_plot` renders scalar fields as Plotly heatmaps:

```python
temperature_plot(
    sin(x) * cos(y),
    (x, -pi, pi),
    (y, -pi, pi),
    samples=(200, 120),
    style={"colorscale": "Viridis", "showscale": True},
)
```

Temperature heatmaps and filled domains use Plotly `zsmooth` by default so low
sample counts read as interpolated fields instead of visibly blocky cells.
Contour lines and domain boundaries use Plotly line smoothing by default. Pass
`style={"zsmooth": False}` for temperature plots, `style={"line_smoothing": 0}`
for contour plots, or nested domain style values to show the raw sampled grid.

`contour_plot` renders scalar fields as Plotly contours:

```python
contour_plot(
    x**2 + y**2,
    (x, -2, 2),
    (y, -2, 2),
    style={"contour_color": "black", "contour_width": 2},
)
```

Free symbols other than the declared independent variables become per-plot
sliders. Explicit slider metadata still goes through `params=`.
Later value or metadata edits are figure-level because parameter state is
shared by the figure:

```python
fig.params = {a: 2.0}
fig.params = {a: {"value": 2.0, "min": 0.5, "max": 6.0, "step": 0.25}}
```

## Authored info

Figures can show authored Markdown notes in the sidebar:

```python
fig = figure("info-demo")
fig.show()
fig.plot(a * sin(x), x, params={a: 1.0})
fig.info("Current amplitude: ", a, name="summary", title="Live info")
```

Strings are literal Markdown. SymPy expressions create and reuse figure
parameters just like plot expressions, so slider edits update the rendered
info. Callable fragments receive the figure and are useful for small computed
diagnostics:

```python
fig.info(
    "Amplitude parameter: ",
    lambda current: current.params[a]["value"],
    name="diagnostic",
)
```

Named info cards replace the existing card with that name, and
`fig.info.clear()` removes authored info without affecting plots. Authored
info is separate from transient status messages such as disconnected-output
notices.

## Domains

`domain_plot` accepts symbolic Boolean conditions, signed scalar expressions,
or finite systems. Signed expressions are inside where they are greater than
zero. Systems are implicit conjunctions.

```python
domain_plot(x**2 + y**2 < 1, x, y)
domain_plot(1 - x**2 - y**2, x, y)
domain_plot([x > 0, y > 0, 1 - x - y], x, y)
```

The fill and boundary are styled independently:

```python
domain_plot(
    1 - x**2 - y**2,
    x,
    y,
    boundary=True,
    style={
        "domain": {"color": "royalblue", "opacity": 0.25, "zsmooth": "best"},
        "boundary": {"color": "royalblue", "width": 2, "smoothing": 1.0},
    },
)
```

Plain Python booleans are rejected because they usually mean Python evaluated
a comparison before SymPy could build a symbolic condition.

## Parametric curves

`parametric_plot` samples a declared parameter interval uniformly. It does not
resample from Cartesian pan or zoom events.

```python
parametric_plot((cos(t), sin(t)), (t, 0, 2*pi))
parametric_plot(Matrix([cos(t), sin(t)]), (t, 0, 2*pi), samples=1000)
```

Only two real coordinate expressions are supported in this phase.

## Figures

`figure("main")` returns or creates a named figure without changing the current
figure. `set_current_figure("main")` changes persistent routing. Figure
contexts route plots temporarily:

```python
with figure("main"):
    plot(sin(x), x)
```

An explicit `figure=` argument routes only one call:

```python
contour_plot(x**2 + y**2, x, y, figure="main")
```

`plot(...)` and the other plot commands display a figure automatically only
until that figure has a visible output. Later plot commands mutate the
existing visible widget, even when they run in later notebook executions.
Returning a figure as the last object in a notebook cell uses IPython's
implicit display hook; that hook is ignored after the figure is already
visible.

`FigureHandle.show()` is the public explicit redisplay operation. It creates a
fresh Jupyter widget display generation, hydrates it from the durable figure
model, and disconnects the previous live generation. `PlotHandle.show()`
delegates to the containing figure.

`fig.view("name")`, `fig.view.current`, `fig.view.range`,
`fig.view.home_range`, `fig.view.x_range`, `fig.view.y_range`,
`fig.view.home_x_range`, `fig.view.home_y_range`, and `fig.view.reset(...)`
group Cartesian view commands under the figure that owns the view state. Range
mutation is intentionally property-based on `fig.view`; lower-level set helpers
are implementation details used by those setters.

### View reset implementation

Python owns both the visible range and the home range. A home range set before
the first display also becomes the initial visible range, so the first Plotly
render opens at the intended home rectangle. After display, changing
`fig.view.home_range` updates the reset target without panning an already
visible plot.

The renderer mirrors the Python-owned home range into Plotly's frontend reset
cache so the modebar reset button and `fig.view.reset()` agree. That bridge is
only synchronization glue; it does not define the source of truth for sampling
or view state.

## Notebook smoke

Manual smoke target: run the examples in JupyterLab or Notebook 7+. Confirm
that automatic display appears once for an undisplayed figure, later plot
commands update that output in place, implicit figure display does not
redisplay an already visible figure, explicit `show()` creates a fresh live
output, old visible outputs say they are disconnected, sliders resample curve,
field, domain, and parametric data, x-axis pan or zoom resamples
non-parametric plots, y-axis pan or zoom resamples fields and domains, domain
fill and boundary styles update independently, and parametric plots keep their
declared parameter samples when the Cartesian viewport changes.

## Lifecycle invariants

The plotting runtime is deliberately one-way:

```text
public call or widget event
    -> source model signal
    -> immutable computed snapshot
    -> focused effect
    -> Plotly or ipywidgets mutation
```

Trace data, trace style, control layout, and slider values use separate
snapshots. The sampled-data renderer receives only sampled arrays, so style
and label changes do not resend numerical samples. The control reconciler
receives slider identity and metadata separately from live values, so changing
a parameter keeps the existing slider widget alive. Model-originated slider
updates run under an observer guard and are not treated as user input.

Frontend widgets are generation-owned. A display generation owns its Plotly
`FigureWidget`, slider registry, layout shell, status area, and widget/Plotly
callbacks. Durable figure and plot nodes hold mathematical state and sampled
buffers. When a newer generation becomes live, the old generation is marked
disconnected, its controls are disabled, and stale slider or Plotly callbacks
are rejected before they can mutate the model.

Plotly does not provide a documented callback-unregistration API for layout
change callbacks in the tested version. The renderer removes its callbacks
from Plotly's internal callback list during `FigureHandle.close()` as a
best-effort cleanup, and real frontend smoke should still check pan and zoom
behavior.

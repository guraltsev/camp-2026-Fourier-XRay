# Seeing with Waves: Fourier Analysis and X-ray Vision 

What does the function $sin(x^2)$ sound like? Can you hear whether a
graph has corners, or whether it can be drawn without lifting your
pen? Can you hear the shape of a drum? Can you determine what an
object looks like just from its X-rays?

In this course, we explore how complicated signals can be broken down
into simple waves. This is Fourier analysis, where functions are sums
of pure oscillations. We will learn how to extract these components
and how to put them back together. We use this to efficiently encode,
compress, and transmit signals.

But what is a Fourier series? (3Blue1Brown):
https://www.youtube.com/watch?v=r6sGWTCMz2k

In many real situations, we do not get to choose what we measure. What
if we only measure the strength of signals passing through an object
from many different directions, like an X-ray machine does? Fourier
analysis can be used to reconstruct what the object looks like. This
is an example of an inverse problem: starting from measurements and we
try to figure out what caused them. Fourier analysis can help us here
as well. However, we will also see that not all inverse problems can
be easily solved.

Expect computer-aided computation, visual exploration, and conceptual
thinking. We will use simple computer tools for simulations and
experiments, but no prior programming experience is required.

**Prerequisites**

Algebra and basic functions, trigonometric functions, and basic
calculus (integrals). Familiarity with complex numbers and basic
linear algebra will be helpful.

## Deployment maintenance

This repository contains a prebuilt static JupyterLite site. Before pushing a
deployment update, run:

```bash
npm run build
```

The local `build.py` script detects the GitHub repository from the `origin`
remote and patches `jupyter-lite.json` so the Pyodide kernel loads from the
correct GitHub Pages URL. It also rewrites the JupyterLab and notebook
workspace links below.

<!-- jupyterlite-lab-url:start -->
## JupyterLab

Open the notebook environment: [https://guraltsev.github.io/camp-2026-Fourier-XRay/lab/index.html](https://guraltsev.github.io/camp-2026-Fourier-XRay/lab/index.html)

Open a notebook workspace:

- [Polynomial Interpolation Instability](https://guraltsev.github.io/camp-2026-Fourier-XRay/lab/workspaces/001-polynomial_interpolation)
- [Pure Trig And Sounds](https://guraltsev.github.io/camp-2026-Fourier-XRay/lab/workspaces/002-pure_trig_and_sounds)
- [Pure Trig Fourier Matching Game](https://guraltsev.github.io/camp-2026-Fourier-XRay/lab/workspaces/003-pure_trig_fourier_matching_game)
- [Square Wave Fourier Matching Game](https://guraltsev.github.io/camp-2026-Fourier-XRay/lab/workspaces/004-square_wave_fourier_matching_game)
- [Products Of Fourier Modes](https://guraltsev.github.io/camp-2026-Fourier-XRay/lab/workspaces/005-Products_of_fourier_modes)
<!-- jupyterlite-lab-url:end -->

To preview locally:

```bash
npm run build -- --local
npm run serve
```

Run `npm run build` again before pushing if you used `--local`.

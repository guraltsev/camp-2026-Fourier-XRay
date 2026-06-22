# math_toolkit JupyterLite

This repository serves a prebuilt JupyterLite site for `math_toolkit`.

The site is static. It already contains the JupyterLite application, notebooks,
browser extensions, local `math_toolkit` source files, and the self-hosted
Pyodide runtime used by the notebooks.

## Preview locally

From this repository root:

```bash
npm run serve
```

That command only starts a static file server for the checked-in files. It does
not build JupyterLite, Pyodide, Python packages, or notebook assets.

## Update the site

Build updates in the source template repository, then copy the generated
`dist/` contents here and commit them.

The deploy repository is intentionally serve-only so GitHub Pages can publish
the exact static files that were built and checked locally.

# Notebook documentation authoring guide

This is the entry point for writing `math_toolkit` documentation notebooks. It
defines the shared themes for all notebooks and points authors to the specific
guide for the kind of page they are writing.

Use:

- [Library guide](LIBRARY_GUIDE.md) for concise topic pages reached directly
  from `Help(...)`.
- [Tutorial guide](TUTORIAL_GUIDE.md) for general tutorials and mathematical
  worksheets.

The technical contract for notebook locations, `_mt_help` metadata, and link
resolution lives in
[Notebook-Documentation-System.md](../../../docs/Notebook-Documentation-System.md).

## Core principle

A documentation notebook should teach a user how to think and work with the
notation, not how the documentation machinery is implemented.

The notebook is successful when a mathematically curious reader can quickly see:

1. what the notation is for,
2. what the first interesting thing to write with it is,
3. how it composes with adjacent ideas,
4. which statements are general and which examples instantiate them,
5. and where the boundaries or surprises are.

Avoid spending the main prose budget on `_mt_help`, topic routing,
metaclasses, runtime-type bookkeeping, or other help-system internals.

## Choose the page type

**Library notebooks** are first landing pages for a topic. They should be
concise, example-first, mathematically motivated, and easy to skim. Use the
[Library guide](LIBRARY_GUIDE.md).

**General tutorials** are practical how-to pages for a specific class of
toolkit work: manipulating symbolic expressions, composing structural notation
into equations and residuals, producing readable output, plotting data, or
converting symbolic expressions to numerical code. They should be focused
enough that a reader can reuse the pattern in nearby work. Use the
[Tutorial guide](TUTORIAL_GUIDE.md).

**Mathematical worksheets** are longer experiences organized by a mathematical
investigation. They may combine symbolic notation, plotting, numerical code,
display, and diagnostics, but the through-line is the mathematical problem
rather than one toolkit capability. Use the
[Tutorial guide](TUTORIAL_GUIDE.md).

When a notebook feels like both, decide by reader intent:

- If the reader arrived to understand one notation feature, write a library
  notebook.
- If the reader arrived to learn a reusable task pattern, write a general
  tutorial.
- If the reader arrived to follow a full mathematical investigation, write a
  mathematical worksheet.

## Shared setup

Documentation notebooks should run in a normal installed-package notebook
session. Do not make the reader depend on repo checkout layout, `.root` files,
or `sys.path` edits.

Use this setup at the start of every self-contained runnable sequence:

```python
import math_toolkit
math_toolkit.notebook.activate()
```

`activate()` injects `md`, so notebooks can use `md("""...""")` for
authored Markdown and mathematical output without importing display helpers.

Examples under `examples/` may include repo-local setup when they are intended
to run directly from a checkout, but shipped documentation notebooks under
`src/math_toolkit/documentation/` should use the installed-package setup.

## Shared writing rules

Always:

- Lead with mathematical payoff, not implementation structure.
- Use short cells with visible results.
- Pair behavioral claims with executable cells when a notebook can demonstrate
  the claim.
- Distinguish general theory from specific examples. When a running example
  specializes a formula, definition, or workflow, make the specialization
  explicit with `subs(...)`, `replace(...)`, a parameter choice, or clear
  prose.
- Show at least one example that composes the topic with another idea.
- Let prose, displayed mathematics, and code names use the same user-facing
  vocabulary. Avoid a separate translation layer of implementation-only names
  unless it is locally needed to compute or inspect a mathematical object.
- Give visible outputs mathematical work to do. Do not display a result only to
  restate names, labels, or implementation bookkeeping already visible in the
  source.
- Explain boundaries in user terms.
- Link to adjacent library pages and tutorials when they exist.
- Keep wording direct and concrete.

Avoid:

- starting with documentation plumbing or runtime type routing,
- large setup cells that mutate import paths in shipped docs,
- several unrelated expressions in one code cell with only the last one shown,
- raw `display(...)`, `Math(...)`, or `Markdown(...)` calls when `md` can
  render the authored output,
- trivial examples whose only purpose is to classify a runtime type,
- examples that silently replace the general idea they were meant to
  instantiate,
- and empty outputs when the output carries the point of the example.

## Output policy

Documentation notebooks should feel alive when opened.

- Commit outputs when they are important to understanding the example.
- Make rendered math visible where that is part of the feature payoff.
- Prefer `md("""...""")` for authored output. It displays Markdown
  automatically.
- Use `display=False` when code needs the rendered string object without
  display.
- Use `{{ expression }}` to insert values into `md` templates.
- Use `{table(header=["name", "value"])}{{ rows }}` for compact reports.

## Review questions

Before merging a notebook change, reviewers should be able to answer "yes" to
the relevant questions:

1. Does the first screen show mathematical payoff?
2. Is there at least one visible, interesting output?
3. Does the page match the appropriate library or tutorial guide?
4. Are semantic claims stated in user terms rather than implementation terms?
5. Does the page distinguish general statements from specific examples?
6. Are examples meaningful enough to show why the notation exists?
7. Would a reader learn how to use the notation, not just how it is classified?

## Authoring markers

When you encounter `@LLM`, `TODO`, `FIXME`, or similar authoring markers,
leave them in place unless the user explicitly asks you to remove them. At the
end of the task, signal any remaining markers and briefly say whether you acted
on them.

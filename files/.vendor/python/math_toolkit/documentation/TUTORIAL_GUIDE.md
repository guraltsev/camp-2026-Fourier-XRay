# Tutorial and worksheet authoring guide

Narrative notebooks are guided learning paths and mathematical showcases. They
help a reader follow a workflow, develop a mathematical idea, or explore how
several notation features work together, with Python-backed computation and
verification making the claims inspectable.

Use the general [authoring guide](AUTHORING_GUIDE.md) for shared principles,
setup, output policy, and review questions.

## Page shapes

**General tutorials** answer practical questions such as "How do I plot data?",
"How do I manipulate symbolic expressions?", or "How do I convert symbolic
expressions to numerical code?" They should be focused enough that a reader can
reuse the pattern in nearby work.

**Mathematical worksheets** are longer experiences organized by a mathematical
investigation. They may combine symbolic notation, plotting, numerical code,
display, and diagnostics, but the through-line is the mathematical problem
rather than one toolkit capability.

Do not group tutorials by a strict split between notation and workflow. In this
toolkit, notation normally exists to support workflows, and workflows normally
teach notation in context.

## Goal

A narrative notebook should answer:

- What mathematical situation are we studying?
- What question, construction, comparison, or workflow organizes the notebook?
- What is the general theory, and what is the running example?
- What sequence of ideas should the reader try?
- How do the pieces fit together in real symbolic work?
- Which claims can be checked by computation, simplification, substitution, or
  numerical experiment?
- What can the reader reuse in nearby problems?

General tutorials should be narrative enough to carry a workflow and concise
enough to finish in one sitting. Mathematical worksheets may be longer when the
mathematical investigation needs more setup, computation, comparison, or
diagnostics. Neither shape is an API reference stub.

At the end of a narrative notebook, a reader should have:

- learned a mathematical topic,
- seen how to model it symbolically or numerically,
- watched important claims become executable checks,
- and understood the relationship between the general theory and the running
  examples.

## Tutorial shapes

Different mathematical subjects need different narrative structures. Choose the
shape that fits the reader's task.

**Guided construction.** Build an object step by step: a recurrence, operator,
basis, map, approximation, diagram, identity, or family of examples.

**Worked analysis.** Start from a mathematical object and examine its behavior:
simplify it, differentiate it, compare cases, compute residuals, inspect a
limit, or test a property.

**Comparison.** Put two or more formulations, algorithms, definitions, or
examples beside each other and show what changes.

**Introspection.** Work in one natural mathematical language, then inspect
specific pieces when the question requires it. For example, treat a system of
equations as a system, while still being able to display one component equation,
one residual, or one substituted instance.

**General theory and specific examples.** Use `subs(...)`, `replace(...)`,
named specializations, or parameter choices to compare a general formula with
concrete instances. Pair these moves with `md` output that shows both the
general statement and the special case.

**Exploration.** Let the reader vary assumptions, parameters, or examples while
keeping a clear mathematical question in view.

These shapes can be combined, but a notebook should have one dominant reason
for existing. For general tutorials, that reason is a reusable task pattern.
For worksheets, that reason is the mathematical through-line.

## Recommended structure

A strong general tutorial usually has:

0. **Setup cell**
   - short activation and imports.
1. **Title and orientation**
    - the organizing question or task,
    - examples of mathematical settings where this is useful
    - the general idea and the running example,
    - the expected payoff.
2. **Incremental mathematical steps**
    - one meaningful move per cell when practical,
    - visible output after each important step.
4. **A composed example or mini-workflow**
   - enough depth to show the notation working across ideas.
5. **A short recap**
    - the toolkit feature that made the work readable,
    - a nearby problem the reader could try next.
6. **Related topics**
    - links back to relevant library pages and adjacent narrative pages.

A strong mathematical worksheet usually has:

0. **Setup cell**
   - short activation and imports.
1. **Title and orientation**
   - the mathematical problem,
   - the organizing investigation,
   - the expected evidence or diagnostic.
3. **Mathematical development**
   - definitions, derivations, specializations, or transformations that move
     the investigation forward.
4. **Computation as evidence**
   - symbolic checks, numerical experiments, plots, or diagnostics near the
     claims they support.
5. **Comparison or diagnosis**
   - what the computation reveals about the mathematical question.
6. **A short recap**
   - the mathematical lesson and the reusable toolkit patterns that supported
     it.
7. **Related topics**
   - links to relevant library pages, tutorials, and worksheets.

3. **General statement and specialization**
    - the general formula, definition, or workflow,
    - the substitutions, replacements, assumptions, or parameter choices that
     produce the running example.

## Representative examples

A tutorial may use one friendly example to teach a broader class of ideas. This
is often the right move in mathematics: a concrete case gives the reader a
place to stand while the notebook points beyond that case.

When using a representative example:

- State what is special about the example and what is meant to generalize.
- Keep the narration centered on the broader method, question, or phenomenon.
- Use special-case simplifications only when they clarify the idea being taught.
- Avoid making the special trick look like the general method.
- Include at least one sentence, comparison, or small variation that shows how
  the workflow adapts outside the running example.

A tractable representative example can keep formulas readable, but the
narrative should not imply that a simplifying feature of the example is the
main theorem unless the tutorial is specifically about that feature.

## General Theory and Specific Examples

The distinction between general theory and specific examples should be visible
throughout a tutorial. A reader should know when they are looking at a general
definition, theorem, construction, or workflow, and when they are looking at a
chosen instance used to make the idea concrete.

Good patterns:

- State the general object before specializing it.
- Create the running example through an explicit `>> Subs({...})`, `replace(...)`,
  parameter choice, assumption, or named specialization.
- Use `>> md` output to show the general formula and the specialized instance
  near each other when the comparison matters.
- Say what the example makes simpler, and whether that simplification is
  accidental or part of the general method.
- Return to the general language after a computation when the lesson is meant
  to transfer.

Avoid letting the example quietly become the theory. If a special case has an
extra symmetry, closed form, cancellation, or linear structure, say so before
the reader mistakes it for the general situation.

## Mathematical through-line

Every tutorial needs a through-line: the idea that lets the reader feel why the
next cell belongs.

Good through-lines include:

- constructing an object and then using it,
- comparing two definitions or algorithms,
- specializing a general statement and comparing it with the instance,
- tracking what changes under substitution, differentiation, simplification, or
  discretization,
- testing whether a property survives a transformation,
- working with one mathematical object while inspecting selected components,
- varying assumptions or parameters and observing the result,
- comparing a general statement with a specific instance.

Name the through-line early. It does not need to be an invariant or diagnostic;
it may be a construction goal, a comparison, a normal form, a geometric picture,
an approximation question, or a reusable symbolic workflow.

## Cell design

Break longer workflows into short worksheet cells. Each cell should have a
mathematical role the reader can name before reading the code. A good tutorial
cell usually does one of these:

- introduces a mathematical object,
- rewrites or transforms an object,
- compares two related objects,
- computes a residual, difference, limit, derivative, component, or
  substitution,
- displays a small report that interprets the result,
- or closes a section with a visible conclusion.

Avoid hiding several conceptual moves in one code cell. If a cell defines the
problem, performs the transformation, computes the comparison, and reports the
conclusion, split it.

Do not hide a necessary mathematical move behind prose. When the notebook
changes the kind of object being studied, show the bridge: derivative to
finite difference, continuous law to recurrence, general formula to
specialization, implicit equation to residual, or exact identity to diagnostic.

Every name introduced in tutorial code should earn its place. Use a name when
it denotes a mathematical object, a recurring diagnostic, or an intermediate
that the prose or output interprets. Avoid aliases whose only job is to shorten
syntax or describe implementation state. If $z_n$ and $z_{n+1}$ are the
notation being taught, use `z[n]` and `z[n + 1]` directly; introduce names such
as `z_midpoint`, `residual`, or `energy_drift` only when those quantities are
mathematical objects in the worksheet.

Every visible output should carry mathematical information. A displayed cell
may introduce a definition, derive an equation, compare cases, verify a claim,
or summarize a conclusion. Do not display output only to say that notation,
labels, or time indices exist.

## Computation and Verification

A finished tutorial should use code as mathematical evidence. The reader
should see not only an answer, but also the definitions, transformations, and
checks that make the answer trustworthy.

- Keep display math attached to the sentence it completes. Do not insert a
  blank line before a display equation when the prose continues the same
  paragraph after the equation.
- When a tutorial compares cases, compute each case in a small visible block or
  report before summarizing the comparison.
- Show intermediate mathematical steps when they carry the lesson. Definitions,
  substitutions, expansions, residuals, invariants, simplified forms, and
  numerical checks should be visible near the claim they support.
- Match Python names to displayed mathematical names when practical. Code
  should read like a continuation of the notation rather than a separate
  translation layer of convenience variables.
- Keep verification proportional to the claim: use exact symbolic checks for
  identities, residuals, and algebraic structure; use numerical experiments for
  behavior that is empirical, approximate, or parameter-dependent.
- Rerun the notebook after edits so execution counts, outputs, and errors
  describe the current source.

## Primary Language and Introspection

Choose one primary mathematical language for the tutorial and stay with it
unless there is a strong reason to introduce another. Multiple competing
representations can make a notebook feel less mathematical, especially when
the reader has to decide whether a scalar equation, vector equation, matrix
equation, or code object is the "real" object.

The toolkit should make the chosen language inspectable. A tutorial can present
a system as a vector equation or a system of equations, then inspect individual
component equations, residuals, substitutions, rows, entries, or examples
without changing the conceptual frame.

Good patterns:

- Work with a system as the main object, then show one equation when its
  structure matters.
- Define a map or operator as the main object, then inspect a component or
  special input.
- Keep a Hamiltonian, recurrence, transformation, or approximation in its
  natural notation, then use substitution or expansion to reveal a local fact.
- Keep the general object and the running example in the same primary language
  whenever possible.
- Use held notation to preserve compact displayed names while still allowing
  expansion when the algebra matters.

Introduce another language only when it serves the tutorial's mathematical
question. If a matrix form, coordinate form, scalar component, or numerical
array appears, explain it as a local inspection or computational tool, not as a
replacement for the primary object.

## Comparisons

Comparison is useful when it clarifies meaning. A short contrast can prevent a
successful computation from looking automatic.

Good comparisons include:

- exact vs approximate,
- preserving vs non-preserving,
- symbolic vs numerical,
- whole object vs selected component,
- direct definition vs transformed form,
- simple case vs perturbed case,
- general expression vs representative example.

Keep comparisons proportional. Their job is to sharpen the main idea, not to
take over the notebook.

## Example quality bar

A tutorial should contain at least one example that would be hard to justify in
plain reference docs, such as:

- a short recurrence model,
- a structured residual for a discrete scheme,
- a system of indexed equations with component-level inspection,
- a family of maps `F[t](x)` followed by substitution or differentiation,
- a comparison between two mathematical encodings,
- a symbolic check of a transformation,
- or a parameterized example that invites exploration.

The point is not to be long. The point is to let the user experience why the
system is pleasant to use.

## Style rules

- Explain steps in plain mathematical language.
- Prefer collaborative or mathematical agency in tutorial prose: use "we" for
  shared exploration or let the mathematical object do the acting. Avoid
  command-style imperatives except when a tutorial is explicitly posing an
  exercise. This convention is specific to tutorials; library docs may still
  use concise imperative guidance.
- State the mathematical relation directly. Avoid stock signposting such as
  "the worksheet question is now", "the distinction matters", "the honest
  object", and similar meta-commentary.
- Describe what a formula contributes to the worksheet, not what the code
  stores. Remove prose or output whose only content is that notation was kept
  visible.
- Use headings that reflect the workflow, not internal package structure.
- Choose one primary mathematical language and use inspection to reveal details.
- End sections with an actual result, not only prose.
- Prefer authored `md("""...""")` reports over raw display piles.
- Keep code readable enough that the mathematics, not the syntax, carries the
  attention.
- Avoid turning tutorials into a second API reference.

## Checklist

A tutorial notebook is ready when all relevant statements are true:

- The first screen names the mathematical setting and payoff.
- The setup is short and does not dominate the opening.
- The notebook has a clear through-line.
- The notebook has a primary mathematical language rather than competing
  representations.
- The general theory and running example are both explicit.
- Specializations are shown with clear substitutions, replacements, assumptions,
  or parameter choices.
- Each section performs a recognizable mathematical move.
- Important claims have nearby executable examples.
- The running example is connected to a broader method or idea when the title
  promises one.
- Special-case simplifications do not masquerade as general techniques.
- Outputs are visible and interpreted.
- The recap names both the mathematical lesson and the reusable workflow.
- Related library pages are linked.


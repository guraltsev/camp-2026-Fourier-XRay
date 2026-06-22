# Library notebook authoring guide

Library notebooks are concise topic pages reached directly from `Help(...)`.
They are the first landing page for a notation feature, object, or small family
of related behaviors.

See the general [authoring guide](AUTHORING_GUIDE.md) for shared principles, setup, output policy, and review questions.

## Goal

A library notebook should answer:

- What is this topic for?
- What is the first interesting thing I can write with it?
- What is the general behavior, and what is merely an example of it?
- How does it relate to neighboring topics?
- What are the important boundaries or pitfalls?

Library notebooks should be concise but not sterile. They are not miniature
implementation notes, and they should not become long tutorials.

## First screen

A reader arriving from `Help(...)` should see all of these without scrolling
far:

- the topic name,
- a one-sentence payoff statement,
- a small runnable example,
- and an output worth looking at.

The setup cell may appear before the title, but it should stay short enough
that the topic name, payoff, first example, and output appear quickly.

The first code example should create a mathematical object or perform a useful
operation, not merely show a bare name.

When the first example specializes a more general idea, name the general idea
first and make the specialization clear. A concise library page can still teach
the reader which facts are part of the topic and which facts belong only to the
chosen example. Use `subs(...)`, `replace(...)`, or a named parameter choice
when that makes the specialization visible without adding much machinery.

## Standard structure

Library notebooks use a frozen vocabulary of level-2 headings. Do not add,
rename, or replace these `##` headings case by case. If a topic needs a custom
label, put that label inside one of the allowed sections as prose, a bold
lead-in, or a `###` subsection where appropriate.

Use only the sections that have real teaching work to do. Omit optional
sections when they would add filler.

1. **Setup cell**
   - first code cell in the notebook,
   - imports and activates `math_toolkit`,
   - no examples or visible output.
2. **Header**
   - topic name,
   - one-sentence payoff,
   - one or two compact usage forms when helpful.
3. **Quick payoff example**
   - one short runnable cell,
   - visible output,
   - no `##` heading.
4. **Core behavior**
   - the primary thing the reader can do with the topic,
   - the first working mental model for reading the notation in ordinary use.
5. **Common patterns**
   - compact recurring idioms and usage forms.
6. **Options and Details**
   - advanced use, alternatives, option sets, modes, flags, and display
     controls.
7. **Semantics**
   - mathematical or logical things that are true about the topic,
   - user-visible guarantees, distinctions, equivalences, and limits.
8. **Examples and Applications**
   - worked examples and compact applications that compose the topic with
     adjacent ideas.
9. **Pitfalls**
   - wrong assumptions, confusing cases, or unsupported expectations.
10. **See also**
    - links to adjacent library pages and at least one relevant tutorial or
      worksheet when one exists.

The only allowed library notebook `##` headings are:

- `## Core behavior`
- `## Common patterns`
- `## Options and Details`
- `## Semantics`
- `## Examples and Applications`
- `## Pitfalls`
- `## See also`

The setup cell, header, quick payoff example, and `## Core behavior` are the
default minimum.

## Heading hierarchy

Use heading levels to describe structure, not merely to label the next example.

- Use `#` only for the topic name in the header cell.
- Use `##` only for the frozen section names.
- Prefer a bold lead-in for a short labeled block inside a section:
  `**Default form.** Use ...`
- In `## Examples and Applications`, use descriptive `### Example: ...`
  subsections before longer `### Application: ...` subsections.
- Outside `## Examples and Applications`, use `###` only when the nested topic
  has enough substance to behave like a real subsection.

## Section guidance

### Common patterns

Use `Common patterns` for compact, reusable idioms. It should help a reader
recognize shapes they will write repeatedly.

Good material includes:

- frequent call shapes or notation forms,
- one or two common variations that clarify argument order or common choices,
- short one-cell examples of ordinary usage,
- and light composition when the point is still the idiom itself.

If a block needs a problem statement, several paragraphs of motivation,
multiple neighboring functions, or several cells, move it to
`Examples and Applications`.

### Options and Details

Use `Options and Details` for advanced information after the common forms are
clear. It should answer "What else can I write or configure?" rather than
"What is true?" or "What complete task can I accomplish?"

Good material includes:

- alternative syntax that is useful but not primary,
- advanced options, modes, flags, and display controls,
- accepted specification forms or compact catalogs of choices,
- and short representative cells.

If an option changes a mathematical or logical guarantee, explain that
guarantee in `Semantics`.

### Semantics

Use `Semantics` for mathematical and logical things that are true about the
topic:

- guarantees or invariants that follow from the notation,
- equivalences or distinctions users may reasonably confuse,
- limits that affect symbolic manipulation,
- distinctions between general behavior and example-specific behavior,
- and neighboring topics a reader may want instead.

Short executable cells are welcome when they demonstrate a truth or boundary.
Do not use `Semantics` for option catalogs, advanced syntax tours, or worked
applications.

Use `### Boundaries` when limitations, non-goals, or neighboring topics need a
skimmable subsection. If the boundary is only a sentence or two, keep it as
prose or a bold lead-in.

Runtime-type mapping may be mentioned only when it clarifies a user-visible
boundary, and then only briefly.

### Examples and Applications

Use `Examples and Applications` for worked material whose purpose is
mathematical payoff rather than syntax recognition, option coverage, or
behavioral rules.

Examples should instantiate the topic without redefining it. When an example is
special because of a chosen parameter, assumption, dimension, index, or
function, say so briefly and keep the general topic visible.

Use descriptive `###` subsections in this order:

1. `### Example: ...` for short focused pieces.
2. `### Application: ...` for longer examples that combine the topic with
   adjacent functions or notation to accomplish something meaningful.

Examples should be shorter than applications. A good example may be one or two
cells that show an identity, transformation, equation, or local composition. A
good application may use several cells and other `math_toolkit` features, but
it should still stay smaller than a tutorial.

## Tone

Good:

- "Use `F[t]` when the function itself varies with a label such as time."
- "Applied forms like `F[t](x)` behave like expressions and can be substituted,
  differentiated, or placed inside equations."

Avoid:

- "Indexed function heads are documented here because the runtime type maps to
  the broader topic."
- "Undefined function heads are documented through a type-level `_mt_help`
  declaration."

## Checklist

A library notebook is ready when all relevant statements are true:

- It opens with a real payoff statement.
- Its first example is mathematically meaningful.
- Substantive notation topics contain at least two worked examples beyond first
  contact.
- Substantive notation topics contain at least one composition example.
- `Common patterns`, when present, stays compact and idiom-focused.
- `Options and Details`, when present, covers advanced use without becoming a
  tutorial.
- `Semantics`, when present, explains mathematical and logical truths in user
  terms.
- Examples distinguish topic-level behavior from facts about the chosen
  instance.
- `Examples and Applications`, when present, uses descriptive subsections.
- Its `##` headings use only the frozen library section names.
- It does not spend significant space on help-system internals.
- It links to at least one relevant tutorial or worksheet when one exists.

## Authoring markers

When you encounter `@LLM`, `TODO`, `FIXME`, or similar authoring markers,
leave them in place unless the user explicitly asks you to remove them. At the
end of the task, signal any remaining markers and briefly say whether you acted
on them.

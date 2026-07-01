# CaML-OP Thesis — LaTeX

Converted from `CaML_OP_Thesis_Final__3_.docx`.

## Files
- `thesis.tex`  — master file (preamble, title page, document structure). Compile this one.
- `front.tex`   — front matter (Statutory Declaration, Acknowledgements, Abstract, List of Abbreviations)
- `main.tex`    — chapters 1–7 (body)
- `refs.tex`    — references
- `media/`      — 14 figures (image1.png … image14.png)
- `thesis.pdf`  — pre-compiled output (57 pages)

## How to compile
Run pdflatex three times (so the TOC, List of Figures, List of Tables,
and cross-references all resolve):

    pdflatex thesis.tex
    pdflatex thesis.tex
    pdflatex thesis.tex

Or with latexmk (recommended — handles the passes automatically):

    latexmk -pdf thesis.tex

Works with TeX Live / MiKTeX / Overleaf. On Overleaf, upload the whole
folder and set `thesis.tex` as the main document.

## What was done in the conversion
- Built a proper title page and `report`-class structure with running headers.
- Front-matter sections are unnumbered chapters; body uses auto-numbered
  chapters/sections/subsections (the manual "1.", "1.1" numbers were stripped
  so LaTeX numbers them).
- The manual Table of Contents / List of Figures / List of Tables were replaced
  with automatic `\tableofcontents`, `\listoffigures`, `\listoftables`.
- All 14 figures wrapped in float environments with auto-numbered captions.
- All 13 body tables given `\captionof{table}` captions; numbering forced to
  match the original (2.1, 4.1–4.3, 5.1–5.9).
- Fixed a styling bug carried over from the .docx where the section
  "2.4 Explainability and the EU AI Act" had a paragraph wrongly marked as the
  heading — restored as a normal heading + body paragraph.

## Things you may want to do next
- The math is currently plain text from the Word file (e.g. `tau-hat = argmin_tau ...`).
  If you want real typeset equations, those passages can be rewritten with proper
  LaTeX math (`\hat{\tau}`, `\sum`, etc.). Tell me and I can convert them.
- References are a plain list. If you have a `.bib` file, this can be switched to
  BibTeX/biblatex for automatic citation formatting.
- The default font is Computer Modern. Swap in another (e.g. lmodern, times)
  by editing the preamble.

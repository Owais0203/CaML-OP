CaML-OP - ICBINB-BIO 2026 submission package

Submission source files in the package root:
  main.tex
  references.bib
  architecture.png
  sign_accuracy.png
  calibration_sweep.png

Final actions and status:
  FINAL_ACTIONS.docx
  CHANGE_LOG.md

Reference-only files:
  reference/CONTENT_MASTER.docx
  reference/CONTENT_MASTER_REFERENCE.pdf

IMPORTANT: use the official, unmodified NeurIPS 2026 style file from the workshop template.
Place the official neurips_2026.sty in the same directory as main.tex before compiling.
Do not substitute an older NeurIPS style file and do not edit the style file.

The submission source is already configured for double-blind workshop review:
  \usepackage[dblblindworkshop]{neurips_2026}
  \workshoptitle{I Can't Believe It's Not Better: Failure Modes of AI in Biology}

Do not add the final or preprint option for the review submission.

ANONYMITY WARNING: do NOT upload this package as-is. Only the five submission source files
listed above are submission artifacts. FINAL_ACTIONS.docx, CHANGE_LOG.md, make_figures.py,
README_FIRST.txt, FORMAT_VERIFICATION.txt and everything under reference/ and style/ are
internal working files. FINAL_ACTIONS.docx contains a personal name and would break
double-blind review if it reached a reviewer. Build the upload from main.tex, references.bib
and the three PNGs plus the official style file, and nothing else.

STATUS: the three P0 experiments in FINAL_ACTIONS.docx are complete and their results are
in main.tex, the two result figures, and CHANGE_LOG.md. Three quantities listed in the
manuscript update table had no replacement value in the P0 output and are handled as
described in CHANGE_LOG.md rather than carried forward as stale numbers.

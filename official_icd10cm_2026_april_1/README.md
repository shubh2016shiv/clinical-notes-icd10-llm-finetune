# Official ICD-10-CM April 1, 2026 Source Bundle

This directory stores the official ICD-10-CM April 1, 2026 release artifacts
used by the `clinical_note_generation_v3` pipeline.

Why this is tracked in Git:
- It preserves the exact source material used to build retrieval assets.
- It makes the project reproducible on another machine without re-fetching data.
- It shows reviewers and recruiters that the pipeline was grounded in the
  official source bundle rather than ad hoc or synthetic code lists.

Contents:
- Raw code lists in `.txt`
- Schema files in `.xsd`
- Official reference PDFs
- The vendor-provided XML release folder

Derived artifacts built from this source, such as FAISS indexes and rule caches,
are intentionally excluded from Git elsewhere in the repository because they can
be regenerated from this bundle.

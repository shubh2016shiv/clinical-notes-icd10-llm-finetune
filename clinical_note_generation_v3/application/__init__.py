"""
application - orchestration layer of the v3 pipeline.

Contains sub-packages that wire core models, infrastructure, and data together
into end-to-end pipeline stages. Nothing in application/ may be imported by
core/ or infrastructure/.

Sub-packages (built progressively across phases):
  bundle_planner/   - registry + sampler for clinical bundle templates (Phase 2)
  icd_resolution/   - maps bundle condition names to official ICD codes (Phase 3)
  note_generation/  - seeded note generation from bundle + constraints (Phase 5)
  evaluation/       - deterministic gates, support verifier, rubric judge (Phase 6)
  pipeline/         - main orchestrator and factory (Phase 8)
"""

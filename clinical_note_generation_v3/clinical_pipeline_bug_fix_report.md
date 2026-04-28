# Clinical Note Evaluation Pipeline — Bug Fix Report

**System:** `clinical_note_generation_v3`
**Acceptance Rate Before Fixes:** 0% (0/10 notes accepted)
**Root Cause Category:** Evaluation gate bugs, not note generation quality

---

## Executive Summary

The pipeline's 0% acceptance rate is caused entirely by two broken evaluation gates. The generated clinical notes are clinically sound. The evaluation infrastructure has three compounding bugs: a false-positive deterministic precheck, an underspecified Gemini response schema that causes the LLM to return empty JSON, and a missing guard that allows empty responses to masquerade as genuine zero scores — triggering hard-fails across all notes.

---

## Bug Index

| # | Bug | File | Failure Share |
|---|---|---|---|
| 1 | ICD description text exact-match false positive | `deterministic_precheck_runner.py` | 40% of notes |
| 2 | `RUBRIC_JUDGE_RESPONSE_SCHEMA` missing inner properties | `contracts.py` | 60% of notes |
| 3 | No guard for hollow LLM response → zeros cascade | `clinical_note_rubric_judge.py` | 60% of notes |
| 4 | Prompt never enumerates criterion key names | `prompt_specs/evaluation.py` | Compounds Bug 2 |
| 5 | `ICD_SELECTION_RESPONSE_SCHEMA` uses unsupported `["string", "null"]` type | `contracts.py` | Latent risk |

---

## Bug 1 — ICD Description False Positive

### Location
`clinical_note_generation_v3/core/services/deterministic_precheck_runner.py` — lines 185–201

### What Is Happening
The precheck performs an exact-match test: if the clinical note contains the literal ICD code description string (e.g. `"adjustment disorder with mixed anxiety and depressed mood"`), the note is rejected for "copying ICD descriptions."

This is a false positive. Clinicians naturally and correctly write diagnostic names in Assessment sections. The precheck cannot distinguish between a lazy verbatim copy of a code description used as a substitute for real documentation, versus a clinician appropriately naming a diagnosis they are actively treating.

### Impact
4 of 10 notes rejected incorrectly. These notes are clinically appropriate.

### The Fix
Replace the blunt exact-match with a context-aware check. Allow diagnostic names when they appear inside expected clinical sections (Assessment, Diagnosis, Impression). Only flag when the ICD description appears in sections where it has no clinical business — such as a verbatim copy in the Plan or HPI with no surrounding clinical context.

```python
# deterministic_precheck_runner.py — replace the ICD description check block

ALLOWED_SECTIONS_FOR_DIAGNOSIS_NAMES = {
    "assessment", "diagnosis", "diagnoses", "impression", "a/p", "assessment and plan"
}

def _is_in_allowed_section(note_text: str, match_start: int) -> bool:
    """
    Walk backwards from match_start to find the nearest section header.
    Return True if that header is an allowed clinical section.
    """
    preceding_text = note_text[:match_start].lower()
    for section_name in ALLOWED_SECTIONS_FOR_DIAGNOSIS_NAMES:
        if section_name in preceding_text:
            # crude proximity check — last occurrence of an allowed header
            last_allowed = preceding_text.rfind(section_name)
            # check no other section header appeared after it
            other_headers = ["plan:", "hpi:", "history of present illness:", "subjective:"]
            if not any(
                preceding_text.rfind(h) > last_allowed
                for h in other_headers
            ):
                return True
    return False

def _check_icd_description_copying(
    self,
    note_text: str,
    icd_descriptions: list[str],
) -> list[str]:
    failures = []
    note_lower = note_text.lower()
    for description in icd_descriptions:
        desc_lower = description.lower()
        match_pos = note_lower.find(desc_lower)
        if match_pos == -1:
            continue  # not present at all — fine
        if _is_in_allowed_section(note_text, match_pos):
            continue  # present in an appropriate section — fine
        failures.append(
            f"ICD description copied verbatim outside a clinical assessment section: '{description}'"
        )
    return failures
```

**If a full section-aware parser is too complex for now**, the minimum viable fix is to whitelist the check entirely for Assessment and Diagnosis sections by skipping the flag when the description appears within N characters after an Assessment header:

```python
ASSESSMENT_HEADERS = ["assessment:", "diagnosis:", "impression:", "a/p:"]

def _icd_copy_is_in_assessment(note_text: str, description: str) -> bool:
    note_lower = note_text.lower()
    desc_lower = description.lower()
    for header in ASSESSMENT_HEADERS:
        header_pos = note_lower.find(header)
        if header_pos == -1:
            continue
        # look for the description within 500 chars after the header
        window = note_lower[header_pos: header_pos + 500]
        if desc_lower in window:
            return True
    return False
```

---

## Bug 2 — `RUBRIC_JUDGE_RESPONSE_SCHEMA` Has No Inner Properties (Primary Root Cause)

### Location
`clinical_note_generation_v3/prompt_specs/contracts.py`

### What Is Happening
The schema passed to Gemini's structured output mode declares the three top-level keys (`general_quality_rubric_scores`, `icd_constraint_alignment_scores`, `icd_constraint_violations`) but defines their contents only as `"type": "object"` or `"type": "array"` with no `properties` specified:

```python
# BROKEN — what the code currently does
RUBRIC_JUDGE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "general_quality_rubric_scores": {"type": "object"},        # ← no properties
        "icd_constraint_alignment_scores": {"type": "object"},      # ← no properties
        "icd_constraint_violations": {"type": "array",
                                      "items": {"type": "object"}}, # ← no properties
    },
}
```

When Gemini's structured output engine sees `"type": "object"` with no `properties`, it has no field names to populate and emits `{}` for those keys. The entire rubric evaluation comes back as three empty containers. This is 100% silent — no exception is raised, no error is logged, `generate_json()` returns a technically valid dict, and the failure propagates downstream as zero scores.

Compare this to `CONDITION_SUPPORT_RESPONSE_SCHEMA` in the same file, which works correctly because it fully enumerates all its properties.

### The Fix
Fully specify every criterion key and its schema. Replace the entire `RUBRIC_JUDGE_RESPONSE_SCHEMA` definition:

```python
# contracts.py — replace RUBRIC_JUDGE_RESPONSE_SCHEMA

_RUBRIC_CRITERION_SCHEMA = {
    "type": "object",
    "properties": {
        "score":     {"type": "integer"},
        "rationale": {"type": "string"},
    },
    "required": ["score", "rationale"],
}

_NULLABLE_RUBRIC_CRITERION_SCHEMA = {
    "type": "object",
    "properties": {
        "score":     {"type": "integer"},
        "rationale": {"type": "string"},
    },
    # intentionally no "required" — Gemini can omit/null these
}

RUBRIC_JUDGE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "general_quality_rubric_scores": {
            "type": "object",
            "properties": {
                "condition_support_coverage":  _RUBRIC_CRITERION_SCHEMA,
                "internal_consistency":        _RUBRIC_CRITERION_SCHEMA,
                "clinical_realism":            _RUBRIC_CRITERION_SCHEMA,
                "encounter_structure_quality": _RUBRIC_CRITERION_SCHEMA,
                "evidence_specificity":        _RUBRIC_CRITERION_SCHEMA,
                "distractor_handling":         _RUBRIC_CRITERION_SCHEMA,
                "assessment_to_plan_linkage":  _RUBRIC_CRITERION_SCHEMA,
                "language_naturalness":        _RUBRIC_CRITERION_SCHEMA,
                "diversity_contribution":      _RUBRIC_CRITERION_SCHEMA,
                "training_utility":            _RUBRIC_CRITERION_SCHEMA,
            },
            "required": [
                "condition_support_coverage", "internal_consistency",
                "clinical_realism", "encounter_structure_quality",
                "evidence_specificity", "distractor_handling",
                "assessment_to_plan_linkage", "language_naturalness",
                "diversity_contribution", "training_utility",
            ],
        },
        "icd_constraint_alignment_scores": {
            "type": "object",
            "properties": {
                "specificity_alignment":               _NULLABLE_RUBRIC_CRITERION_SCHEMA,
                "laterality_alignment":                _NULLABLE_RUBRIC_CRITERION_SCHEMA,
                "encounter_stage_alignment":           _NULLABLE_RUBRIC_CRITERION_SCHEMA,
                "temporal_state_alignment":            _NULLABLE_RUBRIC_CRITERION_SCHEMA,
                "with_without_complication_alignment": _NULLABLE_RUBRIC_CRITERION_SCHEMA,
                "chapter_style_alignment":             _NULLABLE_RUBRIC_CRITERION_SCHEMA,
                "must_not_imply_compliance":           _NULLABLE_RUBRIC_CRITERION_SCHEMA,
            },
            # none required — all are optional/nullable per-case
        },
        "icd_constraint_violations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "violated_constraint_type":            {"type": "string"},
                    "violation_severity":                  {"type": "string"},
                    "what_was_expected":                   {"type": "string"},
                    "what_was_observed_in_note":           {"type": "string"},
                    "fix_instruction_for_revision_prompt": {"type": "string"},
                    "source_icd_code":                     {"type": "string"},
                },
                "required": [
                    "violated_constraint_type", "violation_severity",
                    "what_was_expected", "what_was_observed_in_note",
                    "fix_instruction_for_revision_prompt", "source_icd_code",
                ],
            },
        },
    },
    "required": [
        "general_quality_rubric_scores",
        "icd_constraint_alignment_scores",
        "icd_constraint_violations",
    ],
}
```

---

## Bug 3 — No Guard for Hollow LLM Response (Silent Zero Cascade)

### Location
`clinical_note_generation_v3/application/evaluation/clinical_note_rubric_judge.py` — `judge_generated_clinical_note()` and `_build_single_rubric_criterion()`

### What Is Happening
When `generate_json()` returns `{}` (due to Bug 2), the code has no check for an empty or structurally invalid response. Instead it silently falls through:

```
judgment_response = {}
  → .get("general_quality_rubric_scores", {})  →  {}
    → .get("condition_support_coverage")         →  None
      → _build_single_rubric_criterion(None)     →  score=0, is_missing=False  ← BUG
        → has_any_hard_fail_criterion()           →  True
          → hard_fail_reasons populated
            → final_decision = "reject"
```

The critical flaw: `score=0` from a missing criterion is indistinguishable from `score=0` meaning "this note genuinely failed this criterion." Both look identical to the combiner. The combiner's existing `None` guard (which would produce an honest `"Rubric judging did not produce complete results"` rejection) is never reached because zeros flow through as if they were real evaluations.

### The Fix — Two Parts

**Part A: Add a structural validation guard immediately after `generate_json()`**

```python
# clinical_note_rubric_judge.py — judge_generated_clinical_note()

judgment_response = self._llm_json_generation_client.generate_json(
    judgment_prompt,
    response_schema=prompt_spec.response_schema,
)

# Guard: validate required top-level keys are present and non-empty
_REQUIRED_KEYS = {"general_quality_rubric_scores", "icd_constraint_alignment_scores"}
_missing_or_empty = {
    k for k in _REQUIRED_KEYS
    if not judgment_response.get(k)  # catches both missing and {}
}
if _missing_or_empty:
    import logging
    logging.getLogger(__name__).error(
        "Rubric judge response is missing or empty for keys: %s. "
        "Full response keys returned: %s. "
        "Response snippet: %.800s",
        _missing_or_empty,
        list(judgment_response.keys()),
        str(judgment_response),
    )
    # Return (None, None) to trigger the combiner's honest incomplete-result path
    return (None, None, [], prompt_spec.prompt_id, prompt_spec.prompt_version)
```

The combiner already handles `None` correctly at its existing guard:

```python
# note_quality_decision_combiner.py — this path already exists and works correctly
if general_quality_rubric_scores is None or icd_constraint_alignment_scores is None:
    return NoteEvaluationCritiqueResult(
        ...
        hard_fail_reasons=["Rubric judging did not produce complete results."],
        final_decision="reject",
    )
```

Returning `(None, None)` routes into this honest path instead of the zero-cascade.

**Part B: Add `is_missing` sentinel to `RubricCriterionEvaluation`**

This permanently separates "LLM genuinely scored this 0" from "LLM never returned this criterion":

```python
# core/models/evaluation.py — add field to RubricCriterionEvaluation

@dataclass
class RubricCriterionEvaluation:
    score: int
    rationale: str
    is_missing: bool = False  # True = LLM did not return this criterion at all
```

```python
# clinical_note_rubric_judge.py — _build_single_rubric_criterion()

def _build_single_rubric_criterion(
    self,
    criterion_payload: dict | None,
) -> RubricCriterionEvaluation:
    if not isinstance(criterion_payload, dict):
        return RubricCriterionEvaluation(
            score=0,
            rationale="No rubric evaluation was returned for this criterion.",
            is_missing=True,   # ← sentinel: this is an eval failure, not a quality failure
        )
    return RubricCriterionEvaluation(
        score=int(criterion_payload.get("score", 0)),
        rationale=str(criterion_payload.get("rationale", "")),
        is_missing=False,
    )
```

Then update `has_any_hard_fail_criterion()` in your models to skip missing criteria:

```python
def has_any_hard_fail_criterion(self) -> bool:
    for criterion in self.__dict__.values():
        if isinstance(criterion, RubricCriterionEvaluation):
            # Only hard-fail on genuine zero scores, not on missing evaluations
            if not criterion.is_missing and criterion.score == 0:
                return True
    return False
```

---

## Bug 4 — Prompt Never Enumerates Criterion Key Names

### Location
`clinical_note_generation_v3/prompt_specs/evaluation.py` — `build_rubric_judge_prompt_spec()`

### What Is Happening
The TASK section of the rubric judge prompt tells Gemini to return JSON with `general_quality_rubric_scores` but never lists what fields go inside it. Even without the schema bug, Gemini has no way to know it should produce `condition_support_coverage`, `internal_consistency`, etc. — it can only guess. This compounds Bug 2 and would persist as a source of key-name hallucination even after the schema is fixed.

### The Fix
Replace the vague TASK block with a fully enumerated JSON template:

```python
# prompt_specs/evaluation.py — replace the user_prompt TASK section

    user_prompt = f"""
SEEDED CASE
- Template ID: {seeded_bundle.template_id}
- Archetype: {seeded_bundle.archetype}
- Encounter context: {seeded_bundle.encounter_context}

CONDITION CONSTRAINTS
{chr(10).join(condition_constraint_lines)}

SUPPORT VERIFIER RESULT
- Outcome: {condition_support_verification_outcome.outcome}
- Under-supported conditions: {condition_support_verification_outcome.under_supported_conditions}
- Unsupported implied conditions: {condition_support_verification_outcome.unsupported_implied_conditions}
- History/negation drift: {condition_support_verification_outcome.history_or_negation_drift_detected}
- Notes: {condition_support_verification_outcome.verifier_notes}

GENERATED CLINICAL NOTE
{generated_clinical_note.note_text}

TASK
Return ONLY a JSON object with exactly this structure. Every key shown is required.
Score meanings: 0 = fails criterion, 1 = partially meets, 2 = fully meets.
Use null for icd_constraint_alignment_scores criteria not applicable to this case.
icd_constraint_violations must be [] if there are no violations.

{{
  "general_quality_rubric_scores": {{
    "condition_support_coverage":      {{"score": 0|1|2, "rationale": "..."}},
    "internal_consistency":            {{"score": 0|1|2, "rationale": "..."}},
    "clinical_realism":                {{"score": 0|1|2, "rationale": "..."}},
    "encounter_structure_quality":     {{"score": 0|1|2, "rationale": "..."}},
    "evidence_specificity":            {{"score": 0|1|2, "rationale": "..."}},
    "distractor_handling":             {{"score": 0|1|2, "rationale": "..."}},
    "assessment_to_plan_linkage":      {{"score": 0|1|2, "rationale": "..."}},
    "language_naturalness":            {{"score": 0|1|2, "rationale": "..."}},
    "diversity_contribution":          {{"score": 0|1|2, "rationale": "..."}},
    "training_utility":                {{"score": 0|1|2, "rationale": "..."}}
  }},
  "icd_constraint_alignment_scores": {{
    "specificity_alignment":                   {{"score": 0|1|2, "rationale": "..."}} or null,
    "laterality_alignment":                    {{"score": 0|1|2, "rationale": "..."}} or null,
    "encounter_stage_alignment":               {{"score": 0|1|2, "rationale": "..."}} or null,
    "temporal_state_alignment":                {{"score": 0|1|2, "rationale": "..."}} or null,
    "with_without_complication_alignment":     {{"score": 0|1|2, "rationale": "..."}} or null,
    "chapter_style_alignment":                 {{"score": 0|1|2, "rationale": "..."}} or null,
    "must_not_imply_compliance":               {{"score": 0|1|2, "rationale": "..."}} or null
  }},
  "icd_constraint_violations": [
    {{
      "violated_constraint_type": "...",
      "violation_severity": "advisory|major|critical",
      "what_was_expected": "...",
      "what_was_observed_in_note": "...",
      "fix_instruction_for_revision_prompt": "...",
      "source_icd_code": "..."
    }}
  ]
}}
""".strip()
```

---

## Bug 5 — `ICD_SELECTION_RESPONSE_SCHEMA` Uses Unsupported Gemini Type

### Location
`clinical_note_generation_v3/prompt_specs/contracts.py`

### What Is Happening
```python
# BROKEN
"selected_icd_code":        {"type": ["string", "null"]},
"selected_icd_description": {"type": ["string", "null"]},
```

Gemini's structured output does not support JSON Schema union types (`["string", "null"]`). Passing this schema will cause Gemini to either ignore the constraint silently or produce unexpected output.

### The Fix
```python
# contracts.py — ICD_SELECTION_RESPONSE_SCHEMA

ICD_SELECTION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "selected_icd_code":        {"type": "string"},  # empty string means null
        "selected_icd_description": {"type": "string"},  # empty string means null
        "selection_rationale":      {"type": "string"},
        "resolution_succeeded":     {"type": "boolean"},
    },
    "required": [
        "selected_icd_code",
        "selected_icd_description",
        "selection_rationale",
        "resolution_succeeded",
    ],
}
```

Handle nullability in your Pydantic model or post-processing by treating an empty string as `None`. Pydantic is the correct place for this — not the Gemini schema.

---

## Complete Fix Checklist

```
□  contracts.py
      Replace RUBRIC_JUDGE_RESPONSE_SCHEMA with fully-enumerated schema       (Bug 2)
      Fix ICD_SELECTION_RESPONSE_SCHEMA union type                            (Bug 5)

□  prompt_specs/evaluation.py
      Replace vague TASK block with fully-enumerated JSON template            (Bug 4)

□  clinical_note_rubric_judge.py
      Add structural validation guard after generate_json()                   (Bug 3)
      Add is_missing=True sentinel to _build_single_rubric_criterion()        (Bug 3)

□  core/models/evaluation.py
      Add is_missing: bool = False field to RubricCriterionEvaluation         (Bug 3)
      Update has_any_hard_fail_criterion() to skip is_missing criteria        (Bug 3)

□  deterministic_precheck_runner.py
      Replace exact-match ICD copy check with section-aware check             (Bug 1)
```

---

## Causal Chain — Before and After

### Before (current state)
```
generate_json() called with schema that has no inner properties
    → Gemini emits {} for general_quality_rubric_scores
    → No exception raised, no error logged
    → judgment_response = {"general_quality_rubric_scores": {}, ...}
    → each .get(criterion_name) → None
    → _build_single_rubric_criterion(None) → score=0, is_missing=False
    → has_any_hard_fail_criterion() → True
    → hard_fail_reasons populated
    → final_decision = "reject"   ← wrong reason, wrong path
```

### After (fixed state)
```
generate_json() called with fully-specified schema
    → Gemini emits all 10 criterion keys with real scores
    → structural guard passes (all required keys present)
    → _build_single_rubric_criterion({"score": 2, "rationale": "..."})
    → score=2, is_missing=False
    → has_any_hard_fail_criterion() → False (no genuine zeros)
    → combined_score computed honestly
    → final_decision = "accept" | "revise" | "reject"  ← based on real evaluation
```

---

## Expected Outcome After Fixes

| Metric | Before | After |
|---|---|---|
| Acceptance rate | 0% | Reflects genuine note quality |
| False ICD precheck rejections | 40% of notes | Near 0% for correct clinical language |
| Rubric evaluation failures | 60% of notes | Near 0% (schema drives real output) |
| Reject reason accuracy | Wrong path for all failures | Honest: real quality vs eval infrastructure failure |
| Debuggability | Silent failures | Structured error logs with response snapshots |

# Clinical Note Generation Quality Architecture

## 1. Scope

This document defines the architecture and evaluation workflow for `clinical_note_generation_v3` with one narrow goal:

```text
generate high-quality synthetic clinical notes from pre-resolved clinical condition bundles
```

This document is intentionally not centered on ICD-10 coding accuracy. ICD resolution still exists upstream, but the main evaluation target here is:

```text
clinical note quality
```

That means the main question is:

```text
Does the generated note read like a realistic, useful, internally consistent clinical note for the intended condition bundle?
```


## 2. Problem Statement

If we generate synthetic notes directly from loose profiles such as:

- chronic care
- injury
- pediatric respiratory

then the dataset becomes small-pattern and repetitive.

If we generate notes from raw ICD codes without enough narrative constraints, then the note can become:

- too sparse
- too templated
- too label-explicit
- unrealistic in workflow and tone

So the corrected v3 design should be:

```text
clinical condition bundle
    ->
resolved ICD seed bundle
    ->
clinical note generation
    ->
verifier / critic loop
    ->
quality acceptance or rejection
```

The label bundle is fixed before note generation. The note generator's job is to express that bundle realistically and coherently.


## 3. Architectural Principle

The generator is not deciding the case. It is rendering the case.

The pipeline must preserve this separation:

```text
Bundle planner decides what conditions belong in the case.
Note generator turns that case into realistic text.
Verifier and critic judge the note quality and support coverage.
```

This is important because once note generation is separated from case definition, we can evaluate note quality independently of ICD mapping mistakes.


## 4. High-Level Architecture

```text
                      HIGH-LEVEL V3 ARCHITECTURE

  comorbidity concepts / bundle templates
                  |
                  v
      +---------------------------+
      | Bundle Planner / Sampler  |
      | - archetype               |
      | - encounter context       |
      | - trap pattern            |
      +---------------------------+
                  |
                  v
      +---------------------------+
      | ICD Resolution Layer      |
      | - FAISS/BM25 retrieval    |
      | - constrained code pick   |
      +---------------------------+
                  |
                  v
      +---------------------------+
      | Seeded Bundle             |
      | - active conditions       |
      | - resolved ICD codes      |
      | - traps                   |
      +---------------------------+
                  |
                  v
      +---------------------------+
      | Clinical Note Generator   |
      | - fake PHI                |
      | - realistic structure     |
      | - evidence for conditions |
      +---------------------------+
                  |
                  v
      +---------------------------+
      | Verifier / Critic Loop    |
      | - support coverage        |
      | - realism critique        |
      | - contradiction checks    |
      +---------------------------+
                  |
         +--------+--------+
         |                 |
         v                 v
   +-----------+      +-----------+
   | Accept    |      | Revise /  |
   | example   |      | Reject    |
   +-----------+      +-----------+
```


## 5. Workflow Architecture

### End-to-End Workflow

```text
                  NOTE GENERATION QUALITY WORKFLOW

  Step 1
  Clinical bundle template selected
      |
      v
  Step 2
  Comorbidity concepts normalized
      |
      v
  Step 3
  ICD codes resolved upstream from official ICD data
      |
      v
  Step 4
  Seeded note generation prompt built
      |
      v
  Step 5
  LLM generates synthetic clinical note
      |
      v
  Step 6
  Verifier checks whether the note expresses intended active conditions
      |
      v
  Step 7
  Critic scores realism, coherence, specificity, and note usefulness
      |
      v
  Step 8
  If close to valid: revise once or twice
      |
      v
  Step 9
  Final acceptance / rejection
      |
      v
  Step 10
  Write training artifact + audit artifact
```


## 6. Section-Wise Component Explanation

### 6.1 Bundle Planner

This component defines what the note is supposed to be about.

Its inputs are:

- comorbidity bundle templates
- encounter archetypes
- trap patterns
- complexity tiers

Its outputs are:

- active condition bundle
- encounter context
- allowed distractors

This layer exists so that note generation is guided by a meaningful scenario rather than a vague profile prompt.


### 6.2 ICD Resolution Layer

This layer maps clinical concepts to official ICD codes. It is upstream context for generation, not the main evaluation focus of this document.

Responsibilities:

- normalize clinical condition names
- retrieve official code candidates
- select a stable resolved seed bundle

Output:

```text
resolved seed bundle
```

Example:

```text
type 2 diabetes mellitus -> E11.9
essential hypertension  -> I10
hyperlipidemia          -> E78.5 or E78.2
```


### 6.2.1 Official ICD Metadata as a Clinical Constraint Source

The official ICD data should not be used only for code resolution.

The FAISS metadata derived from the official ICD source also contains semantic signals that can make generated clinical notes more reliable:

- `description`
- `short_description`
- `chapter_prefix`
- `is_billable`

These fields can be converted into note-generation constraints.


### 6.2.2 What Semantic Nuances Can Be Extracted

From the official descriptions, we can reliably extract patterns such as:

- `with`
- `without`
- `unspecified`
- `initial encounter`
- `subsequent encounter`
- `sequela`
- `right`
- `left`
- `bilateral`
- `acute`
- `chronic`
- `recurrent`
- `in remission`
- `history`

These are not just coding details. They are narrative obligations for the note.


### 6.2.3 How These Nuances Improve Clinical Note Quality

Each of these patterns can constrain the note generator:

- `with`
  - the note must include evidence for the linked complication or associated condition

- `without`
  - the note must avoid implying the excluded complication

- `unspecified`
  - the note should avoid accidentally supporting a more specific subtype unless intended

- `initial encounter`
  - the note should read like a fresh presentation, not a follow-up

- `subsequent encounter`
  - the note should include follow-up or healing context

- `sequela`
  - the note should describe residual effects rather than acute presentation

- `right`, `left`, `bilateral`
  - the note must state laterality clearly

- `acute`, `chronic`, `recurrent`
  - the note should match the time-course implied by the code

- `in remission`, `history`
  - the note must avoid making the condition sound actively symptomatic unless intended


### 6.2.4 Chapter Prefix as Encounter-Style Guidance

`chapter_prefix` is also useful as a note-style control signal.

Examples:

- endocrine / metabolic families
  - chronic management, labs, medication adherence, follow-up planning

- injury families
  - mechanism, site, laterality, encounter timing, imaging, acute management

- behavioral health families
  - symptom severity, duration, functional impact, safety review, treatment follow-up

- respiratory / infectious families
  - symptom onset, exam findings, imaging or testing context, uncertainty around organism or severity

This means the official ICD metadata can influence not just what is written, but how the note is structured.


### 6.2.5 Required New Layer: Code Semantic Constraint Extractor

The architecture should include a component that converts resolved ICD codes plus official metadata into structured note-generation constraints.

Suggested output:

```json
{
  "seed_code": "S52.502A",
  "description": "Unspecified fracture of the lower end of left radius, initial encounter for closed fracture",
  "constraints": {
    "laterality": "left",
    "encounter_type": "initial",
    "acuity": "acute",
    "body_site": "lower end of radius",
    "must_include": [
      "acute injury presentation",
      "left-sided symptoms or exam findings",
      "fracture confirmation or imaging evidence",
      "initial treatment plan"
    ],
    "must_not_imply": [
      "right-sided injury",
      "follow-up healing visit",
      "residual sequela unless intended"
    ]
  }
}
```

This layer makes the generated notes more reliable because the model is not just given a label. It is given the semantic obligations implied by that label.


### 6.2.5A Implementation Rule: Tiered Constraint Extraction

The constraint extractor should not be implemented as regex-only or LLM-only.

Use a tiered strategy:

1. deterministic pattern matcher first
2. family-specific rule pack second
3. LLM escalation only for unresolved or ambiguous cases

The deterministic matcher should cover high-confidence signals such as:

```python
PATTERNS = {
    "laterality": r"\\b(right|left|bilateral)\\b",
    "encounter": r"\\b(initial encounter|subsequent encounter|sequela)\\b",
    "temporal": r"\\b(acute|chronic|recurrent|in remission)\\b",
    "with_flag": r"\\bwith\\b",
    "without_flag": r"\\bwithout\\b",
    "unspecified": r"\\bunspecified\\b",
}
```

Escalate to an LLM only when:

- multiple qualifiers conflict
- a family-specific description is compound or ambiguous
- severity and temporal cues must be interpreted together
- no deterministic rule matches but a known family requires semantic parsing

Every extracted constraint should carry:

- `value`
- `confidence`
- `source`
- `matched_span`

This keeps extraction cheap for common cases and debuggable for edge cases.


### 6.2.5B Implementation Rule: Family-Specific Validation Before Scaling

Do not assume one extractor works across all chapters.

Before bulk generation, validate the extractor on at least one worked example from each representative family:

- infectious / respiratory
- injury
- endocrine / metabolic
- behavioral health
- neoplasm

These examples should become a permanent regression suite for extractor behavior and note-evaluation logic.


### 6.2.6 How to Capture Metadata Nuances for Evaluation

The same semantic constraints used during note generation should also be carried into evaluation.

For each seeded code, the pipeline should create a structured evaluation payload such as:

```json
{
  "seed_code": "F33.1",
  "description": "Major depressive disorder, recurrent, moderate",
  "applicable_constraints": {
    "laterality": null,
    "encounter_type": null,
    "temporal_state": ["recurrent"],
    "severity": ["moderate"],
    "must_include": [
      "current depressive symptoms",
      "recurrent history or prior episodes"
    ],
    "must_not_imply": [
      "psychotic features unless intentionally seeded",
      "full remission"
    ]
  }
}
```

This payload becomes input to:

- the support verifier
- the rubric judge
- deterministic mismatch checks

That way, nuance extraction is not only helping generation. It is also helping evaluation stay code-aware without becoming code-accuracy obsessed.


### 6.3 Seeded Clinical Note Generator

This is the main component for this document.

Its job is to write a note that:

- clearly supports the seeded active conditions
- includes realistic clinical evidence
- includes workflow structure
- uses fake PHI if desired
- includes natural distractors
- does not leak ICD codes into note text

Expected note content:

- chief complaint or visit reason
- HPI / interval history
- relevant ROS
- exam or observational findings
- labs / imaging when appropriate
- assessment
- plan

The note should not read like a code description expanded into prose.


### 6.4 Verifier

The verifier answers a narrow question:

```text
Does the generated note actually support the intended active conditions?
```

The verifier can use:

- retrieval
- constrained selector
- direct critique prompt
- deterministic checks

The verifier is not primarily scoring label correctness here. It is checking whether the note text contains enough support and does not drift.


### 6.5 Critic

The critic evaluates note quality beyond condition support.

This component should detect:

- unrealistic note structure
- contradictory narrative
- templated language
- missing assessment-to-plan linkage
- weak or absent evidence
- excessive direct copying of official descriptions

The critic should return structured feedback so the system can revise notes when the issues are fixable.


### 6.5.1 Critic Stack

The critic should be implemented as a layered evaluation stack instead of one vague LLM judgment.

```text
generated note
    ->
deterministic pre-checks
    ->
support verifier
    ->
rubric-based LLM judge
    ->
revision decision
```

Each layer has a different job:

- deterministic checks catch obvious failures cheaply
- support verifier checks whether intended conditions are actually expressed
- rubric judge evaluates realism, coherence, and training usefulness


### 6.5.2 Deterministic Pre-Checks

These should run before expensive critique.

Suggested checks:

- note is non-empty and above minimum length
- note does not contain literal ICD code strings
- note contains required structural components
- note is not an exact or near-exact duplicate of recent accepted notes
- note does not overuse repeated boilerplate phrases
- note does not copy official ICD descriptions too literally


### 6.5.2A Implementation Rule: Deterministic Diversity Gate

Do not ask the LLM judge to score diversity contribution as the primary defense against duplication.

Use a deterministic near-duplicate gate before rubric judging:

```python
embedding = embed(candidate_note)
similarity = max(cosine(embedding, prior_embedding) for prior_embedding in recent_accepted_embeddings)
if similarity > 0.92:
    return hard_fail("near_duplicate")
```

Recommended implementation:

- use a rolling window of recent accepted notes
- combine embedding similarity with one cheap lexical overlap or section-template overlap check
- treat near-duplicate detection as a binary gate, not a soft rubric score

Output:

- `pass`
- `hard_fail`

If a note fails here, it should not enter the LLM critique stage.


### 6.5.3 Support Verifier

The support verifier evaluates whether the note adequately expresses the intended active condition bundle.

It can use:

- retrieval
- constrained selector
- direct support-check prompting

Its job is to detect:

- under-supported intended conditions
- unsupported extra implied conditions
- history-only or negated drift
- contradiction with the intended bundle

This is still part of note-quality control, because a realistic-sounding note that weakly expresses the case is poor training data.


### 6.5.4 Rubric-Based LLM Judge

After deterministic checks and support verification, a rubric-based LLM judge should score the note.

This is the best place to use a framework such as `DeepEval`, because the evaluation is criteria-based and subjective in a controlled way.

Recommended role for `DeepEval` or equivalent custom judging:

- clinical realism
- internal consistency
- encounter structure quality
- evidence specificity
- distractor handling
- assessment-to-plan linkage
- language naturalness
- training utility

Optional role for `Ragas`-style metrics:

- secondary faithfulness / relevancy signals when retrieval context is involved

The LLM judge should return:

- criterion scores
- short rationale per criterion
- revision guidance
- final recommendation: `accept`, `revise`, or `reject`


### 6.5.5 Structured Critique Output

The critic output should be structured rather than free-form.

Suggested schema:

```json
{
  "deterministic_precheck_status": "pass",
  "support_verifier_status": "pass",
  "rubric_scores": {
    "condition_support_coverage": 2,
    "internal_consistency": 2,
    "clinical_realism": 1,
    "encounter_structure_quality": 2,
    "evidence_specificity": 1,
    "distractor_handling": 2,
    "assessment_to_plan_linkage": 1,
    "language_naturalness": 1,
    "diversity_contribution": 1,
    "training_utility": 2
  },
  "hard_fail_reasons": [],
  "revision_targets": [
    "make lab and exam support more specific",
    "reduce templated assessment phrasing"
  ],
  "final_decision": "revise"
}
```

This gives the revision loop something concrete to act on.


### 6.6 Revision Loop

The revision loop exists because some notes are close to good but not yet acceptable.

Use:

- `1` initial generation
- `1-2` revision attempts max

This prevents infinite prompt churn and keeps the pipeline measurable.

Revision is allowed only when the note is near-valid. If the note is fundamentally wrong or too artificial, reject it.


### 6.6.1 Iterative Improvement Policy

The revision loop should be fixed and measurable:

1. generate initial note
2. run deterministic pre-checks
3. run support verifier
4. run rubric judge
5. if result is `accept`, keep note
6. if result is `revise`, send only targeted critique back to the generator
7. rerun evaluation
8. if still weak after max revisions, reject


### 6.6.2 What Triggers Revision

Revision is appropriate when:

- intended conditions are present but weakly evidenced
- note is plausible but too generic
- assessment and plan are only partially linked
- distractor wording is awkward but fixable
- language is too templated but clinically coherent


### 6.6.3 What Triggers Immediate Rejection

Reject without revision when:

- the note contradicts core intended conditions
- one or more intended conditions are missing entirely
- the note introduces strong unsupported active diagnoses
- the note leaks ICD codes directly
- the note is structurally broken or too repetitive
- the note reads like code-description paraphrase instead of a note


### 6.6.4 Revision Prompting Rule

The revision step should receive:

- the original seed bundle
- the previous note
- only the structured critique items that need fixing

The revision step must not redefine the case. The case stays fixed.


### 6.6.4A Implementation Rule: Freeze the Bundle in Revision

The revision call should place the seed bundle and semantic constraints in read-only system context, and place only the revision targets in the user message.

The revision prompt must explicitly forbid changing:

- active conditions
- encounter type
- laterality
- temporal state
- patient context unless the critique explicitly requires a local clarification


### 6.6.4B Implementation Rule: Post-Revision Drift Check

After every revision, run a cheap drift check before reconsidering the note for acceptance.

At minimum verify that revision did not change:

- recovered active ICD bundle
- laterality
- encounter stage
- temporal state
- major patient context

If revision fixes one issue but causes drift on a previously valid constraint, reject rather than accept.


### 6.7 Artifact Writing

Accepted examples should produce:

1. a training row
2. an audit row

The training row should stay minimal:

```json
{
  "clinical_note": "...",
  "icd10_codes": [...]
}
```

The audit row should be richer:

- source bundle
- generation prompt metadata
- generated note
- verifier output
- critic output
- revision history
- acceptance decision


## 7. Why the Verifier Loop Is Needed

Without a verifier loop, the note generator will often produce notes that are plausible but weak for training.

Typical failure patterns:

- one intended condition is under-expressed
- an unintended active condition becomes implied
- history-only distractors look active
- the note becomes too generic
- the note sounds templated across many cases

So the loop is necessary because the real target is not merely:

```text
synthetic note that looks okay
```

It is:

```text
synthetic note that is realistic, information-rich, internally coherent, and aligned to the intended condition bundle
```


## 8. Evaluation Philosophy

For this stage of v3, the evaluation target is:

```text
clinical note generation quality
```

not:

```text
perfect ICD coding accuracy
```

That means the rubric should judge notes on:

- realism
- coherence
- evidentiary support
- diversity
- usefulness as supervision text

The rubric should not be dominated by whether the note maps to the single best ICD subtype.


## 9. Evaluation Execution Plan

The rubric is only useful if it is executed consistently.

Recommended execution order:

```text
note
    ->
deterministic pre-checks
    ->
support verifier
    ->
rubric-based LLM judge
    ->
accept / revise / reject
```


### 9.1 Deterministic Gate

This gate removes clearly bad notes cheaply.

Outputs:

- `pass`
- `hard_fail`


### 9.2 Support Verification Gate

This gate answers:

```text
Does the note adequately express the intended active bundle?
```

This should be treated as a required gate, not merely a soft score.


### 9.3 Rubric Scoring Gate

This gate scores note quality using the rubric below.

This is the best place to apply `DeepEval`-style rubric scoring or equivalent custom judge logic.


### 9.4 Decision Gate

The final gate combines:

- deterministic outcomes
- support verifier result
- rubric scores

Final result:

- `accept`
- `revise`
- `reject`


### 9.4A Implementation Rule: Constraint Severity Taxonomy

Metadata constraints must be classified before they can influence acceptance consistently.

Use three severity levels:

- `critical`
  - violations force hard fail
- `important`
  - violations usually trigger revision
- `advisory`
  - violations inform scoring but do not block acceptance alone

Recommended defaults:

- laterality mismatch in injury-style notes: `critical`
- encounter-stage mismatch in encounter-sensitive codes: `critical`
- with/without contradiction: `critical`
- temporal-state mismatch for remission/history/recurrent codes: `important` or `critical` depending on family
- chapter-style weakness: `important`
- stylistic awkwardness: `advisory`


## 10. Evaluation Rubric

Each note should be scored against a structured rubric. Use a `0-2` scale per criterion:

- `0` = fail
- `1` = partial / weak
- `2` = strong

Total score is the sum of all criteria.


### 10.0 Rubric Structure

The rubric should have two layers:

1. `general note-quality rubric`
2. `metadata-alignment rubric`

The first judges whether the note is a good clinical note in general.

The second judges whether the note respects the semantic obligations implied by the seeded codes and official metadata.

This makes evaluation more reliable because a note can be realistic overall while still violating code-driven nuance such as laterality, encounter timing, or remission state.


### 10.0A Implementation Rule: Explicit Score Combination

The scoring formula must be explicit.

Use:

```text
final_score =
  0.6 * normalized_general_quality_score
  + 0.4 * normalized_metadata_alignment_score
```

Apply hard-fail rules before computing final acceptance:

- deterministic pre-check fail
- support verifier fail
- critical metadata constraint fail

This prevents ambiguous interpretation of raw score totals when different notes have different numbers of applicable metadata constraints.


### 10.1 Condition Support Coverage

Question:

```text
Does the note provide enough textual evidence for each intended active condition?
```

Scoring:

- `0`: one or more seeded conditions are not meaningfully supported
- `1`: all seeded conditions appear, but one or more are weakly supported
- `2`: every seeded condition is clearly and naturally supported


### 10.1A Metadata Constraint Alignment

Question:

```text
Does the note respect the semantic constraints derived from the seeded codes and official ICD metadata?
```

This is a grouped criterion composed of applicable subchecks. Only applicable subchecks should be scored for a given note.


### 10.1A.1 Specificity Alignment

Question:

```text
Does the note support the seeded specificity level without accidentally implying a different subtype?
```

Examples:

- `unspecified` codes should not accidentally become strongly specific
- specific codes should have enough detail to justify their specificity

Scoring:

- `0`: clearly mismatched specificity
- `1`: partially aligned but ambiguous
- `2`: specificity matches cleanly


### 10.1A.2 Laterality Alignment

Question:

```text
If the seeded code implies right, left, or bilateral involvement, does the note match it clearly?
```

Scoring:

- `0`: mismatched or missing laterality
- `1`: present but weak or slightly ambiguous
- `2`: clearly aligned


### 10.1A.3 Encounter-Stage Alignment

Question:

```text
If the code implies initial encounter, subsequent encounter, or sequela, does the note narrative match that stage?
```

Scoring:

- `0`: encounter stage clearly wrong
- `1`: partially aligned but not fully convincing
- `2`: narrative clearly matches the stage


### 10.1A.4 Temporal-State Alignment

Question:

```text
If the code implies acute, chronic, recurrent, remission, or history state, does the note reflect that time-course correctly?
```

Scoring:

- `0`: wrong temporal state
- `1`: partly aligned but weak
- `2`: clearly aligned


### 10.1A.5 With/Without Complication Alignment

Question:

```text
If the code includes 'with' or 'without' semantics, does the note include or exclude the corresponding complication appropriately?
```

Scoring:

- `0`: contradiction of with/without semantics
- `1`: weak or incomplete alignment
- `2`: clean alignment


### 10.1A.6 Chapter-Style Alignment

Question:

```text
Does the note structure and evidence style match the encounter expectations implied by the code family or chapter?
```

Examples:

- injury notes should include mechanism, site, laterality, timing, and acute management
- chronic endocrine notes should include monitoring, meds, labs, or follow-up logic
- behavioral notes should include symptoms, severity, functioning, or safety context

Scoring:

- `0`: encounter style clearly mismatched
- `1`: partly aligned
- `2`: strongly aligned


### 10.1A.7 Must-Not-Imply Compliance

Question:

```text
Does the note avoid implying things that the semantic constraint extractor marked as forbidden drift?
```

Scoring:

- `0`: forbidden drift is present
- `1`: slight risk of drift
- `2`: no meaningful drift


### 10.2 Internal Consistency

Question:

```text
Is the note internally coherent, without contradictions between history, exam, assessment, and plan?
```

Scoring:

- `0`: clear contradictions or implausible flow
- `1`: minor inconsistency or weak transition
- `2`: consistent throughout


### 10.3 Clinical Realism

Question:

```text
Does the note resemble a plausible real-world clinical note for the encounter type?
```

Scoring:

- `0`: reads synthetic or artificial
- `1`: mostly plausible but generic or uneven
- `2`: convincingly realistic


### 10.4 Encounter Structure Quality

Question:

```text
Does the note contain a believable clinical workflow structure?
```

Scoring:

- `0`: missing major structural components
- `1`: present but thin or mechanically assembled
- `2`: coherent structure with natural progression


### 10.5 Evidence Specificity

Question:

```text
Are the diagnoses supported by note-specific evidence rather than generic statements?
```

Scoring:

- `0`: mostly label restatement
- `1`: some evidence but weak specificity
- `2`: specific findings, symptoms, labs, imaging, or management details support the note


### 10.6 Distractor Handling

Question:

```text
Are negated, historical, or uncertain conditions included in a controlled and believable way?
```

Scoring:

- `0`: distractors are confusing or look active
- `1`: distractors are present but awkward
- `2`: distractors are natural and clearly separated from active conditions


### 10.7 Assessment-to-Plan Linkage

Question:

```text
Do the plan elements logically follow from the assessment?
```

Scoring:

- `0`: plan does not match assessment
- `1`: partially linked
- `2`: plan naturally follows the active issues


### 10.8 Language Naturalness

Question:

```text
Does the prose sound like a clinician note rather than a paraphrased ontology entry?
```

Scoring:

- `0`: robotic, repetitive, or copied from code descriptions
- `1`: understandable but stiff
- `2`: natural note language


### 10.9 Diversity Contribution

Question:

```text
Does this note add meaningful variety to the dataset in phrasing, structure, or encounter style?
```

Scoring:

- `0`: near-template duplicate
- `1`: some variation but limited
- `2`: meaningfully distinct from recent examples


### 10.10 Training Utility

Question:

```text
Would this note be useful as supervision text for a model learning note-to-code mapping?
```

Scoring:

- `0`: weak, noisy, or misleading training signal
- `1`: usable with caveats
- `2`: strong training example


### 10.11 Normalized Scoring for Applicable Constraints

Not every note will have:

- laterality constraints
- encounter-stage constraints
- with/without semantics
- remission/history semantics

So metadata-alignment scoring should be normalized over only the applicable subcriteria.

Example:

- if a note has no laterality-sensitive code, skip laterality alignment
- if a note has no encounter-stage-sensitive code, skip encounter-stage alignment

This prevents unfair scoring across very different note types.


### 10.12 Implementation Rule: Critique Payload Must Carry Metadata Violations

The revision loop must receive metadata-alignment failures in machine-readable form.

Extend the critique payload with:

```json
{
  "metadata_alignment_scores": {
    "laterality_alignment": 0,
    "encounter_stage_alignment": 2
  },
  "metadata_violations": [
    {
      "constraint_type": "laterality",
      "severity": "critical",
      "expected": "left",
      "observed": "right wrist pain",
      "fix_instruction": "change findings and assessment to left-sided involvement only"
    }
  ]
}
```

Without this, the revision loop cannot fix metadata nuance errors consistently.


## 11. Aggregate Acceptance Policy

Suggested acceptance thresholds:

- hard fail if `Condition Support Coverage = 0`
- hard fail if `Metadata Constraint Alignment = 0` for any applicable critical constraint
- hard fail if `Internal Consistency = 0`
- hard fail if `Clinical Realism = 0`
- hard fail if `Training Utility = 0`
- hard fail if deterministic pre-checks fail
- hard fail if support verifier fails

Then use total score thresholds:

- `17-20`: accept
- `13-16`: revise once
- `9-12`: revise only if the critique is localized
- `< 9`: reject

This policy is intentionally quality-first.

The thresholds should be interpreted over:

- general note-quality criteria
- normalized metadata-alignment criteria

not as a blind sum over irrelevant fields.


### 11.1 Revision Cutoff Logic

Suggested loop policy:

- max revisions: `2`
- if score does not improve after a revision: reject
- if the same criterion remains `0` twice: reject
- if revision introduces new contradictions: reject


## 12. Metrics to Track Across a Run

Do not evaluate notes one by one only. Track batch-level quality metrics:

- acceptance rate
- average rubric score
- average score by encounter archetype
- average score by bundle template
- deterministic hard-fail rate
- support-verifier fail rate
- revision rate
- revision success rate
- rejection reasons
- template-collapse rate
- note leakage rate

These metrics will tell us whether the pipeline is scaling gracefully or just producing a few good examples by chance.


### 12.1 Metrics for Iterative Improvement

To improve prompt and model behavior over time, track:

- mean rubric score by generator prompt version
- mean rubric score by model
- pass rate after first generation
- pass rate after first revision
- pass rate after second revision
- most frequent failing rubric criteria
- most frequent deterministic failures
- most frequent support-verifier failures


## 13. Recommended First Implementation Focus

The first implementation focus should be:

1. `bundle template registry`
2. `seeded note generation prompt`
3. `code semantic constraint extractor`
4. `deterministic pre-check schema`
5. `support verifier schema`
6. `critic rubric schema`
7. `accept / revise / reject loop`
8. `batch-level quality metrics`


### 13.1 Implementation Rule: Minimal Bundle Template Registry Schema

The bundle template registry should start as flat JSON or JSONL files, not a database or service.

Use a minimal schema:

```json
{
  "template_id": "chronic_metabolic_01",
  "archetype": "chronic_care_followup",
  "complexity_tier": 2,
  "active_conditions": [
    "type 2 diabetes mellitus",
    "essential hypertension",
    "hyperlipidemia"
  ],
  "encounter_context": "outpatient follow-up",
  "allowed_distractors": [
    "history of smoking",
    "resolved UTI"
  ],
  "trap_patterns": [
    "do not imply diabetic nephropathy unless seeded"
  ]
}
```

This is enough to stabilize bundle planning without premature infrastructure.

This order is correct because note quality depends more on:

- case design
- metadata-derived semantic constraints
- generation constraints
- evaluation structure

than on deeper orchestration technology.


## 14. Final Position

The clearest way to think about `clinical_note_generation_v3` is:

```text
It is a clinical note quality pipeline with upstream label control.
```

The ICD bundle defines what the note must express.

The note generator is judged primarily on:

- realism
- support coverage
- coherence
- usefulness as supervision text

That is the right evaluation target if the immediate goal is to produce stronger synthetic clinical notes before sweating ICD subtype precision.

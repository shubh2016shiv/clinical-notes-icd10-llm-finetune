# Corrected Clinical Note Generation Architecture

## 1. Mistake We Made

The earlier v2 and early v3 direction treated the synthetic clinical note as the starting point:

```text
profile -> generate note -> retrieve ICD candidates -> select ICD codes
```

That is useful for auditing or benchmarking a coder, but it is not the right foundation for building fine-tuning data for:

```text
clinical note -> ICD-10 code prediction
```

If the note is generated first, then the labels are being inferred after the fact. That creates three problems:

1. The note can accidentally support more diagnoses than intended.
2. The selected codes can drift from the real supervision target.
3. The final dataset teaches the model to imitate a selector's interpretation of a note, instead of learning from examples built around deliberate label control.

The corrected objective is:

```text
official ICD-10-CM codes -> generate note that expresses those codes -> verify recovery of those same codes -> keep only validated examples
```


## 2. Correct Objective

The data generator must produce synthetic training rows whose labels are decided before note generation.

The correct pipeline is:

```text
official ICD-10-CM repository
    ->
meaningful ICD seed bundle sampler
    ->
seeded clinical note generator
    ->
retrieval-constrained ICD verifier
    ->
seed recovery validator
    ->
accepted training example
```

In this design:

- The official ICD data is the source of truth.
- The seed ICD codes are the intended labels.
- The generated note is only a textual realization of those labels.
- The selector is only a verifier, not the primary label creator.


## 3. Core Design Principle

The pipeline must enforce this invariant:

```text
The diagnoses are chosen first.
The note is generated second.
The verifier checks whether the note truly supports the chosen diagnoses.
```

This separates responsibilities cleanly:

- `Sampler` decides what the case is about.
- `Generator` expresses that case in note form.
- `Verifier` checks whether the note still matches the seeded labels.
- `Validator` decides whether the example is safe enough to keep.


## 4. Corrected End-to-End Flow

### Step 1: Build a Seed Bundle From Official ICD-10-CM

The starting object is not a profile string. It is a structured seed bundle:

```json
{
  "bundle_id": "bundle_00001",
  "clinical_archetype": "adult_chronic_multimorbidity",
  "seed_codes": [
    {
      "code": "E11.9",
      "description": "Type 2 diabetes mellitus without complications"
    },
    {
      "code": "I10",
      "description": "Essential (primary) hypertension"
    },
    {
      "code": "E78.2",
      "description": "Mixed hyperlipidemia"
    }
  ],
  "trap_conditions": [
    "history of asthma",
    "denies chest pain"
  ],
  "encounter_context": "routine primary care follow-up"
}
```

The seed bundle is the intended ground-truth label set.


### Step 2: Generate the Clinical Note From That Bundle

The generation prompt receives:

- encounter context
- seeded active ICD codes
- official descriptions
- required evidence expectations
- optional negation/history traps
- instruction not to leak ICD codes into note text

The LLM produces:

- `clinical_note`
- `note_summary`
- optional `evidence_map`
- optional `generator_self_check`


### Step 3: Verify the Generated Note

Now the generated note goes through the existing retrieval-constrained verification path:

```text
clinical note
    ->
hybrid retrieval
    ->
candidate ICD list
    ->
selector chooses only from candidates
```

This verifier is useful because it stress-tests whether the note actually contains enough textual evidence for the intended labels.


### Step 4: Validate Seed Recovery

The validator compares:

- `seed_codes`
- `selected_codes`
- note text
- quality flags

The example is accepted only if:

1. Every seeded code is recovered by the verifier.
2. No unsupported extra active diagnosis is selected.
3. All selected codes are official and billable.
4. The note does not contradict seeded diagnoses.
5. Negated or history-only distractors are not selected as active.
6. Quality flags are either empty or explicitly allowed by policy.


### Step 5: Write Final Artifacts

There should be two outputs:

1. **Training artifact**

```json
{
  "clinical_note": "...",
  "icd10_codes": [
    {
      "code": "E11.9",
      "description": "Type 2 diabetes mellitus without complications"
    },
    {
      "code": "I10",
      "description": "Essential (primary) hypertension"
    }
  ]
}
```

2. **Audit artifact**

Contains:

- seed bundle
- generated note
- retrieved candidates
- verifier output
- validation result
- provenance


## 5. Corrected V3 Architecture

### High-Level Architecture

```text
official ICD-10-CM data
    ->
OfficialICDRepository
    ->
ICDSeedSampler
    ->
SeededCodeBundle
    ->
SeededClinicalNoteGenerator
    ->
GeneratedClinicalNoteExample
    ->
HybridICDCandidateRetriever
    ->
ICDSelectorVerifier
    ->
SeedRecoveryValidator
    ->
TrainingArtifactWriter + AuditArtifactWriter
```


### Recommended New Core Objects

#### `SeededCodeBundle`

Responsibilities:

- hold the intended active codes
- hold encounter context
- hold trap conditions
- hold metadata for reproducible sampling

Suggested fields:

```python
bundle_id
clinical_archetype
encounter_context
seed_codes
trap_conditions
sampling_metadata
```


#### `GeneratedClinicalNoteExample`

Responsibilities:

- store the note generated from a seed bundle
- keep note-level metadata before verification

Suggested fields:

```python
seed_bundle
note_summary
clinical_note
generator_evidence_map
generation_warnings
```


#### `SeedRecoveryValidationResult`

Responsibilities:

- compare seed labels with verifier output
- explain why an example should be accepted or rejected

Suggested fields:

```python
is_valid
missing_seed_codes
unsupported_extra_codes
contradicted_seed_codes
historical_or_negated_selection_errors
quality_flags
```


## 6. Most Important Correction: Sampling Cannot Be Profile-Only

If we sample only from 10 to 20 profiles, we will create a small, repetitive dataset. That would make downstream fine-tuning weak and portfolio credibility low.

Profiles should remain only one conditioning axis. They must not be the primary sampling universe.

The true sampling unit is:

```text
meaningful ICD seed bundle
```

not:

```text
profile label
```


## 7. Initial Basis for Meaningful ICD Sampling

This is the most important design choice in the corrected pipeline.

The sampler must be justified, reproducible, and capable of scaling to hundreds or thousands of distinct note-label combinations.

### 7.1 Sampling Unit

Each generated example begins with a bundle of `1-4` active billable ICD codes.

Why `1-4`:

- `1` supports simple cases and sparse notes.
- `2-3` covers common real-world outpatient and ED notes.
- `4` supports moderate multimorbidity without making the note unnatural.
- More than `4` active codes too often turns synthetic notes into crowded, low-clarity supervision.


### 7.2 Sampling Must Be Stratified, Not Uniform Random

Uniform random sampling over 74,719 billable codes will fail. It will generate:

- too many obscure codes
- too many incompatible combinations
- too many notes that are hard to express naturally

Instead, sampling should be stratified across four axes:

1. `clinical archetype`
2. `ICD chapter / code family`
3. `complexity tier`
4. `coverage tier`


### 7.3 Clinical Archetype as a Sampling Axis, Not the Root

Examples:

- adult chronic primary care
- acute urgent care injury
- behavioral health outpatient
- pediatric respiratory / infectious
- cardiometabolic follow-up
- renal-metabolic follow-up
- musculoskeletal pain / injury
- preventive / annual exam with relevant active conditions

These archetypes control note style and encounter context. They do not decide the labels alone.


### 7.4 ICD Chapter / Code Family Sampling

The official ICD dataset already gives us the structured label space. The sampler should organize billable codes into meaningful families such as:

- diabetes
- hypertension
- hyperlipidemia
- CKD
- asthma
- depression
- anxiety
- UTI
- pneumonia
- fractures
- sprains

This family layer is what makes bundles meaningful.

The sampler should not initially try to cover the entire ICD universe evenly. It should start with code families that:

1. are common enough to support repeated note generation
2. have clear textual evidence patterns
3. have meaningful near-miss alternatives for verification
4. are relevant to a coding benchmark portfolio


### 7.5 Complexity Tier

The sampler should assign each example a complexity tier:

- `Tier 1`: 1 active code, minimal traps
- `Tier 2`: 2 active codes, one distractor
- `Tier 3`: 3 active codes, mixed active/history/negation
- `Tier 4`: 4 active codes, moderate ambiguity, richer note

This creates a progression from easy to realistic.


### 7.6 Coverage Tier

The dataset should be sampled across coverage tiers:

- `head`: common codes and families
- `mid`: moderately common but still documentable
- `tail`: less common but still note-generatable without specialist nonsense

Without this, the model will overfit only to a few high-frequency patterns.


## 8. Recommended Initial Sampling Strategy

### Phase 1: Portfolio-Safe Starter Universe

Start with a curated subset of clinically documentable families rather than the full 74,719-code space.

Recommended initial family groups:

- endocrine / metabolic:
  - diabetes without complications
  - diabetes with common documented complications only when explicitly expressible
  - obesity
  - hyperlipidemia
- cardiovascular:
  - essential hypertension
  - old MI
  - stable chronic ischemic history/status cases where documentation is clear
- renal:
  - CKD stage 1-4
- respiratory:
  - asthma uncomplicated
  - asthma exacerbation
  - pneumonia
  - bronchiolitis
- infectious / genitourinary:
  - UTI
- behavioral health:
  - recurrent MDD
  - GAD
  - alcohol use disorder in remission
- injury / musculoskeletal:
  - distal radius fracture
  - ankle sprain
  - wrist sprain

This is still a subset, but it is a large enough family graph to generate many meaningful combinations.


### Phase 2: Bundle Templates, Not Just Code Lists

The sampler should use bundle templates such as:

- `chronic triad`
  - diabetes + hypertension + hyperlipidemia
- `renal-metabolic`
  - diabetes + CKD + hypertension
- `behavioral + medical comorbidity`
  - MDD + GAD + hypertension
- `single acute injury`
  - fracture only
- `acute injury with chronic comorbidity`
  - sprain + obesity
- `pediatric infection`
  - bronchiolitis only
- `pediatric respiratory mixed`
  - pneumonia + asthma exacerbation

This is justified because clinical notes are usually built around coherent condition constellations, not arbitrary code bags.


### Phase 3: Coverage Quotas

For each run, quotas should be enforced across:

- archetypes
- family groups
- complexity tiers
- head/mid/tail buckets

This avoids collapsing into 80 percent cardiometabolic notes.


## 9. How to Make Sampling Scalable Beyond 10 to 20 Profiles

The scalable unit is not the profile count. It is the cross-product of:

- archetype
- family bundle template
- specific ICD code variant
- complexity tier
- trap pattern
- documentation style variation

Even a moderate design can create large diversity:

```text
12 archetypes
x 25 family bundle templates
x 10 code variants per template
x 4 complexity tiers
x 6 trap patterns
= 7,200 structured generation configurations
```

That is already enough to support a meaningful synthetic fine-tuning set.

This is why the corrected architecture must focus on a seed-bundle library and quota-based stratified sampling rather than a tiny hand-authored profile list.


## 10. Proposed V3 Module Additions

Add these modules without changing the existing README:

```text
clinical_note_generation_v3/
  application/
    note_generation/
      icd_seeded_note_generation_pipeline.py
    prompts/
      icd_seeded_note_generation_prompt.py

  core/
    models/
      seed_bundle.py
    services/
      icd_seed_sampler.py
      seed_recovery_validator.py
```


### `icd_seed_sampler.py`

Responsibilities:

- choose meaningful seed bundles from official ICD data
- enforce family compatibility rules
- enforce quotas
- support deterministic sampling with `random_seed`


### `icd_seeded_note_generation_prompt.py`

Responsibilities:

- build prompt from seed bundle
- instruct model to express the exact active diagnoses
- require evidence, traps, and no code leakage


### `icd_seeded_note_generation_pipeline.py`

Responsibilities:

- orchestrate seed sampling
- generate note from bundle
- call verifier
- call seed recovery validator
- write training and audit artifacts


### `seed_recovery_validator.py`

Responsibilities:

- compare seed bundle to verifier output
- fail examples with drift or unsupported extras


## 11. Acceptance Criteria for the Corrected Generator

The corrected generator is working only when these conditions hold:

1. Every example starts from official ICD billable codes.
2. The note is generated from those seed codes, not from profile-only free generation.
3. The note contains evidence for each seeded active code.
4. The verifier recovers the seeded codes.
5. Extra unsupported active codes cause rejection.
6. Negated and historical distractors are not accepted as active.
7. Final training data stores note-to-code supervision directly.
8. Audit data preserves the full verification trail.


## 12. Implementation Order

Build in this order:

1. `SeededCodeBundle` model
2. `ICDSeedSampler`
3. `icd_seeded_note_generation_prompt.py`
4. `icd_seeded_note_generation_pipeline.py`
5. reuse existing retriever and selector as verifier
6. `SeedRecoveryValidator`
7. `training_example_writer.py` and `audit_example_writer.py`


## 13. Final Decision

The corrected architecture is:

```text
code-first generation with note verification
```

not:

```text
note-first generation with post-hoc label selection
```

That correction keeps the official ICD-10 data in control of the labels, makes the synthetic notes serve the labels rather than define them, and gives the project a far stronger path toward meaningful fine-tuning data.

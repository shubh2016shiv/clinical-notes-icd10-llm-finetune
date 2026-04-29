"""
Centralized prompt specs for note generation and revision.
"""

from __future__ import annotations

from clinical_note_generation_v3.core.models.constraints import (
    ClinicalBundleSemanticConstraints,
)
from clinical_note_generation_v3.prompt_specs.contracts import (
    CLINICAL_NOTE_RESPONSE_SCHEMA,
    JSON_ONLY_OUTPUT_CONTRACT,
    PromptSpec,
)
from clinical_note_generation_v3.prompt_specs.rendering import render_list_block

GENERATION_PROMPT_ID = "clinical_note_generation"
GENERATION_PROMPT_VERSION = "v2_centralized"
REVISION_PROMPT_ID = "clinical_note_revision"
REVISION_PROMPT_VERSION = "v2_centralized"

ANTI_COPY_RULES = [
    "CRITICAL: Never write any ICD-10-CM, CPT, or other alphanumeric billing codes anywhere in the note — not in parentheses after diagnoses, not in problem lists, not in assessments, not anywhere. Clinical notes are written in clinical language only.",
    "Do not copy or near-copy official ICD short or long description phrasing.",
    "Do not write ontology-like prose, code explanations, or label-expansion definitions.",
    "Do not annotate diagnosis names with classification codes. Write the clinical condition in natural language as a clinician would document it.",
]

BAD_STYLE_EXAMPLES = [
    'Bad (ICD code leakage): "Personal history of sigmoid colon cancer (Z85.038)" — write instead: "Personal history of sigmoid colon adenocarcinoma, status post curative resection."',
    'Bad (ICD code leakage): "1. Essential hypertension (I10), well-controlled." — write instead: "1. Essential hypertension, well-controlled on current regimen."',
    'Bad (description copying): "The patient has essential primary hypertension, which is characterized by elevated blood pressure."',
    'Bad (flat plan): "Plan: Address diabetes. Address hypertension. Follow up."',
]


def build_generation_prompt_spec(
    *,
    bundle_semantic_constraints: ClinicalBundleSemanticConstraints,
    fake_patient_name: str,
    fake_patient_medical_record_number: str,
    fake_patient_date_of_birth: str,
    generation_attempt_number: int,
) -> PromptSpec:
    seeded_bundle = bundle_semantic_constraints.seeded_bundle
    system_prompt = f"""
<persona>
  <role>Senior Clinical Documentation Specialist</role>
  <task_frame>You are rendering a fixed, pre-resolved clinical case into a realistic fictional encounter note for synthetic training data generation.</task_frame>
  <identity_constraints>
    <constraint>You are not selecting, inferring, or inventing diagnoses.</constraint>
    <constraint>You are not paraphrasing ICD ontology entries or short descriptions.</constraint>
    <constraint>You are faithfully expressing a fixed case bundle as a clinician would write it during or after an encounter.</constraint>
  </identity_constraints>
</persona>

<objective>
  <primary>Generate a fictional but clinically realistic encounter note where all seeded active conditions are naturally supported, clinically evidenced, internally coherent, and expressed in realistic encounter style.</primary>
  <secondary>Evidence must be encounter-grounded and must not merely restate the condition label.</secondary>
</objective>

<execution_expectations>
  <case_fidelity>
    <rule>The seeded bundle is read-only. Preserve laterality, encounter stage, temporal state, and patient identity exactly.</rule>
  </case_fidelity>
  <clinical_realism>
    <rule>The note must read like practicing clinician documentation, not a coding worksheet.</rule>
    <rule>Write symptoms, findings, history, results, and management first. Let diagnosis phrasing emerge from evidence.</rule>
  </clinical_realism>
  <lexical_discipline>
    {"".join(f"<rule>{rule}</rule>" for rule in ANTI_COPY_RULES)}
  </lexical_discipline>
  <required_structure>
    <rule>The note MUST contain all of the following clearly labeled sections, in order. Missing any section is a hard failure.</rule>
    <section>Chief Complaint (or CC:)</section>
    <section>History of Present Illness (or HPI:)</section>
    <section>Past Medical History (or PMH:)</section>
    <section>Medications</section>
    <section>Allergies</section>
    <section>Physical Examination (or Physical Exam:)</section>
    <section>Assessment</section>
    <section>Plan</section>
  </required_structure>
  <assessment_plan_linkage>
    <rule>Every seeded active condition must be addressed in the Assessment and/or Plan.</rule>
  </assessment_plan_linkage>
</execution_expectations>

<bad_style_examples>
  {"".join(f"<example>{example}</example>" for example in BAD_STYLE_EXAMPLES)}
</bad_style_examples>

<output_contract>
  <format>{JSON_ONLY_OUTPUT_CONTRACT}</format>
  <schema>{{ "clinical_note_text": "<full note text>" }}</schema>
</output_contract>
""".strip()

    user_prompt = f"""
<task>Generate a fictional clinical encounter note from the fixed case bundle below. Return only the JSON output contract.</task>

<case_bundle>
  <template_id>{seeded_bundle.template_id}</template_id>
  <generation_attempt>{generation_attempt_number}</generation_attempt>
  <encounter>
    <archetype>{seeded_bundle.archetype}</archetype>
    <context>{seeded_bundle.encounter_context}</context>
  </encounter>
  <patient>
    <name>{fake_patient_name}</name>
    <mrn>{fake_patient_medical_record_number}</mrn>
    <dob>{fake_patient_date_of_birth}</dob>
  </patient>
  <active_conditions>
{_build_active_condition_rendering(bundle_semantic_constraints)}
  </active_conditions>
  <bundle_requirements>
{_indent_block(_build_bundle_level_requirements(bundle_semantic_constraints), 4)}
  </bundle_requirements>
  <prohibited_implications>
{_indent_block(_build_prohibited_implication_summary(bundle_semantic_constraints), 4)}
  </prohibited_implications>
  <narrative_style>
    {_build_narrative_style_guidance(bundle_semantic_constraints.dominant_chapter_family())}
  </narrative_style>
  <distractor_guidance>
{_indent_block(_build_distractor_guidance(bundle_semantic_constraints), 4)}
  </distractor_guidance>
  <trap_pattern_guidance>
{_indent_block(_build_trap_pattern_guidance(bundle_semantic_constraints), 4)}
  </trap_pattern_guidance>
</case_bundle>

<generation_instructions>
  <step order="1">Internalize all seeded conditions and treat the case bundle as immutable.</step>
  <step order="2">Identify evidence types that would realistically support each active condition in this encounter.</step>
  <step order="3">Draft the note with evidence-forward narrative and realistic workflow structure.</step>
  <step order="4">Verify that every seeded condition appears in Assessment and/or Plan with realistic management language.</step>
  <step order="5">Confirm all required structural sections are present with clear headings: Chief Complaint, HPI, Past Medical History, Medications, Allergies, Physical Examination, Assessment, Plan.</step>
  <step order="6">Check for ICD code leakage, label expansion, ontology prose, and specificity drift before returning JSON.</step>
</generation_instructions>
""".strip()

    return PromptSpec(
        prompt_id=GENERATION_PROMPT_ID,
        prompt_version=GENERATION_PROMPT_VERSION,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_schema=CLINICAL_NOTE_RESPONSE_SCHEMA,
    )


def build_revision_prompt_spec(
    *,
    bundle_semantic_constraints: ClinicalBundleSemanticConstraints,
    previous_clinical_note_text: str,
    revision_targets: list[str],
    metadata_fix_instructions: list[str],
    fake_patient_name: str,
    fake_patient_medical_record_number: str,
    fake_patient_date_of_birth: str,
    generation_attempt_number: int,
) -> PromptSpec:
    seeded_bundle = bundle_semantic_constraints.seeded_bundle
    system_prompt = f"""
<persona>
  <role>Senior Clinical Documentation Specialist - Revision Mode</role>
  <task_frame>You are performing a targeted revision of an existing fictional clinical note. Correct only the identified deficiencies while preserving the frozen case semantics.</task_frame>
</persona>

<revision_freeze_rules>
  <frozen_element>All seeded active conditions</frozen_element>
  <frozen_element>Laterality, encounter stage, temporal state, and patient identity</frozen_element>
  <frozen_element>Major clinical scenario and all non-deficient content</frozen_element>
</revision_freeze_rules>

<lexical_discipline>
  {"".join(f"<rule>{rule}</rule>" for rule in ANTI_COPY_RULES)}
</lexical_discipline>

<revision_scope_rules>
  <rule>Apply only the requested revision targets.</rule>
  <rule>Do not rewrite the note from scratch unless the revision target explicitly requires it.</rule>
  <rule>{JSON_ONLY_OUTPUT_CONTRACT}</rule>
</revision_scope_rules>

<output_contract>
  <schema>{{ "clinical_note_text": "<full revised note text>" }}</schema>
</output_contract>
""".strip()

    combined_targets = revision_targets + metadata_fix_instructions
    if not combined_targets:
        combined_targets = [
            "Improve specificity, realism, and support coverage while preserving the same fixed case."
        ]

    user_prompt = f"""
<task>Perform a targeted revision of the clinical note below. Apply only the specified revision targets. Preserve all frozen case semantics. Return only the JSON output contract with the complete revised note.</task>

<frozen_case_bundle>
  <template_id>{seeded_bundle.template_id}</template_id>
  <encounter>
    <archetype>{seeded_bundle.archetype}</archetype>
    <context>{seeded_bundle.encounter_context}</context>
  </encounter>
  <patient>
    <name>{fake_patient_name}</name>
    <mrn>{fake_patient_medical_record_number}</mrn>
    <dob>{fake_patient_date_of_birth}</dob>
  </patient>
  <active_conditions>
{_build_active_condition_rendering(bundle_semantic_constraints)}
  </active_conditions>
  <prohibited_implications>
{_indent_block(_build_prohibited_implication_summary(bundle_semantic_constraints), 4)}
  </prohibited_implications>
  <trap_patterns>
{_indent_block(_build_trap_pattern_guidance(bundle_semantic_constraints), 4)}
  </trap_patterns>
</frozen_case_bundle>

<previous_note>
{previous_clinical_note_text}
</previous_note>

<revision_targets>
{_indent_block(render_list_block(combined_targets, default_line="- None."), 2)}
</revision_targets>

<revision_instructions>
  <step order="1">Memorize the frozen case bundle, prohibited implications, and trap patterns before modifying the note.</step>
  <step order="2">Process each revision target in order.</step>
  <step order="3">Verify that no frozen element changed, no trap pattern was violated, and no new deficiency was introduced.</step>
  <step order="4">Return the complete revised note as JSON only.</step>
</revision_instructions>

<revision_attempt>{generation_attempt_number}</revision_attempt>
""".strip()

    return PromptSpec(
        prompt_id=REVISION_PROMPT_ID,
        prompt_version=REVISION_PROMPT_VERSION,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_schema=CLINICAL_NOTE_RESPONSE_SCHEMA,
    )


def _build_active_condition_rendering(
    bundle_semantic_constraints: ClinicalBundleSemanticConstraints,
) -> str:
    rendered_conditions: list[str] = []
    for resolved_condition_entry, condition_semantic_constraints in zip(
        bundle_semantic_constraints.seeded_bundle.resolved_conditions,
        bundle_semantic_constraints.per_code_note_writing_constraints,
        strict=False,
    ):
        rendered_conditions.append(
            _indent_block(
                "\n".join(
                    [
                        "<condition>",
                        f"  <label>{resolved_condition_entry.condition_name}</label>",
                        f"  <laterality>{condition_semantic_constraints.laterality or 'N/A'}</laterality>",
                        f"  <encounter_stage>{condition_semantic_constraints.encounter_type or 'N/A'}</encounter_stage>",
                        f"  <temporal_state>{', '.join(condition_semantic_constraints.temporal_states) or 'N/A'}</temporal_state>",
                        render_list_xml(
                            "must_include", condition_semantic_constraints.must_include_in_note
                        ),
                        render_list_xml(
                            "must_not_imply", condition_semantic_constraints.must_not_imply_in_note
                        ),
                        "</condition>",
                    ]
                ),
                4,
            )
        )
    return "\n".join(rendered_conditions)


def _build_bundle_level_requirements(
    bundle_semantic_constraints: ClinicalBundleSemanticConstraints,
) -> str:
    return render_list_block(
        bundle_semantic_constraints.all_must_include_items(),
        default_line="- Support every seeded condition with realistic clinical evidence.",
    )


def _build_prohibited_implication_summary(
    bundle_semantic_constraints: ClinicalBundleSemanticConstraints,
) -> str:
    return render_list_block(
        bundle_semantic_constraints.all_must_not_imply_items(),
        default_line="- Do not add unsupported active diagnoses or contradictory findings.",
    )


def _build_narrative_style_guidance(dominant_chapter_family: str) -> str:
    guidance_map = {
        "injury": "Use an injury-focused note with mechanism, findings, imaging context, and acute management.",
        "endocrine_metabolic": "Use chronic-care follow-up language with labs, medications, monitoring, and longitudinal management.",
        "behavioral_health": "Use behavioral-health documentation tone with symptoms, mental status, function, and follow-up treatment.",
        "respiratory_infectious": "Use acute respiratory/infectious visit language with onset, exam findings, testing, and response to treatment.",
        "neoplasm": "Use oncology-aware documentation with disease status, treatment context, and surveillance language.",
        "other": "Use balanced clinician-authored note style matched to the encounter context.",
    }
    return guidance_map.get(dominant_chapter_family, guidance_map["other"])


def _build_distractor_guidance(
    bundle_semantic_constraints: ClinicalBundleSemanticConstraints,
) -> str:
    return render_list_block(
        [
            f"{distractor} (historical, negated, ruled out, or secondary only)"
            for distractor in bundle_semantic_constraints.seeded_bundle.allowed_distractors
        ],
        default_line="- If secondary details appear, keep them realistic and non-active.",
    )


def _build_trap_pattern_guidance(
    bundle_semantic_constraints: ClinicalBundleSemanticConstraints,
) -> str:
    return render_list_block(
        bundle_semantic_constraints.seeded_bundle.trap_patterns,
        default_line="- Do not drift from the fixed case semantics.",
    )


def render_list_xml(tag_name: str, items: list[str]) -> str:
    if not items:
        return f"  <{tag_name}>None.</{tag_name}>"
    item_lines = "\n".join(f"    <item>{item}</item>" for item in items)
    return f"  <{tag_name}>\n{item_lines}\n  </{tag_name}>"


def _indent_block(text: str, indent: int) -> str:
    padding = " " * indent
    return "\n".join(f"{padding}{line}" if line else "" for line in text.splitlines())

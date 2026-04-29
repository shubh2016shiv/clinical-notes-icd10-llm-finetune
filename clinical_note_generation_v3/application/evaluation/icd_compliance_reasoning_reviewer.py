"""
ICD-10 compliance reasoning reviewer.

Uses a reasoning model (GPT-5 Nano by default) to review a generated clinical
note for ICD-10 coding accuracy in a single LLM call that simultaneously:
  1. Detects compliance issues (duplicate codes, missing documentation, wrong specificity, etc.)
  2. Produces concrete fix instructions for the note generator to act on immediately.

The caller loops up to icd_compliance_reviewer_max_regeneration_loops times:
  generate -> review -> if note_fix_needed: regenerate with fixes -> review -> ...

LAYER: application/evaluation
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IcdComplianceReviewResult:
    """
    Structured output from one ICD-10 compliance review call.

    Fields
    ------
    note_fix_needed
        True when the reviewer found at least one ICD-10 compliance issue.
    fixes
        Ordered list of specific, actionable fix instructions for the note
        generator. Empty when note_fix_needed is False.
    """

    note_fix_needed: bool
    fixes: list[str] = field(default_factory=list)


class IcdComplianceReasoningReviewer:
    """
    Reviews a generated clinical note for ICD-10-CM compliance using a
    reasoning model (GPT-5 Nano).

    A single call to review_icd_compliance checks the note and produces fix
    instructions. The pipeline wraps this in a regeneration loop controlled by
    icd_compliance_reviewer_max_regeneration_loops.

    On any API or parse error the reviewer fails open (note_fix_needed=False)
    so a transient failure never blocks the pipeline.
    """

    provider_name = "openai"

    def __init__(self, *, model_name: str, api_key: str | None = None) -> None:
        import os

        self.model_name = model_name
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for the ICD compliance reasoning reviewer."
            )

        from openai import OpenAI

        self._client = OpenAI(api_key=self._api_key)

    def review_icd_compliance(
        self,
        *,
        clinical_note_text: str,
        seeded_icd_codes: list[str],
        active_conditions: list[str],
        condition_icd_pairs: list[tuple[str, str, str]],
    ) -> IcdComplianceReviewResult:
        """
        Review a clinical note for ICD-10 compliance in a single reasoning call.
        """
        prompt = self._build_review_prompt(
            clinical_note_text=clinical_note_text,
            seeded_icd_codes=seeded_icd_codes,
            active_conditions=active_conditions,
            condition_icd_pairs=condition_icd_pairs,
        )

        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert ICD-10-CM compliance reviewer for clinical documentation. "
                            "Return only valid JSON with no markdown fences. "
                            'Required keys: "note_fix_needed" (boolean) and "fixes" (array of strings).'
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            response_text = response.choices[0].message.content or ""
            parsed = json.loads(response_text)

            note_fix_needed = bool(parsed.get("note_fix_needed", False))
            raw_fixes = parsed.get("fixes", [])
            fixes = [str(fix) for fix in raw_fixes if fix]

            if note_fix_needed and not fixes:
                logger.warning(
                    "ICD compliance reviewer set note_fix_needed=true but returned no fixes; "
                    "treating as compliant to avoid an uninstructed regeneration loop."
                )
                note_fix_needed = False

            return IcdComplianceReviewResult(note_fix_needed=note_fix_needed, fixes=fixes)

        except Exception as review_error:
            logger.warning(
                "ICD compliance reasoning reviewer call failed; skipping review: %s",
                review_error,
            )
            return IcdComplianceReviewResult(note_fix_needed=False, fixes=[])

    @staticmethod
    def _build_review_prompt(
        *,
        clinical_note_text: str,
        seeded_icd_codes: list[str],
        active_conditions: list[str],
        condition_icd_pairs: list[tuple[str, str, str]],
    ) -> str:
        conditions_block = "\n".join(
            f"  - {name}: {code} - {description}" for name, code, description in condition_icd_pairs
        )
        codes_inline = ", ".join(seeded_icd_codes) if seeded_icd_codes else "(none)"
        conditions_inline = ", ".join(active_conditions) if active_conditions else "(none)"

        return f"""\
You are performing an ICD-10-CM compliance audit on a synthetic clinical note \
before it is written to a training dataset.

REQUIRED DIAGNOSES
{conditions_block}

Required ICD-10 codes : {codes_inline}
Active condition names : {conditions_inline}

CLINICAL NOTE
{clinical_note_text}

COMPLIANCE CHECKLIST
Check all five criteria and flag any violation:

1. NO DUPLICATE CODES
   The note must not document the same diagnosis in a way that implies the same
   ICD-10 code more than once (for example, listing J45.41 twice).

2. CODE SPECIFICITY MATCH
   Clinical detail in the note must justify the assigned code; no over-coding
   (documenting complications that would upgrade the code) and no under-coding
   (vague language that does not support the required specificity).

3. ALL CONDITIONS DOCUMENTED
   Every condition in the required list must be clearly described in the note
   narrative with enough clinical detail to support coding. A condition that
   is only mentioned in passing (for example, "history of X" when X is active)
   is insufficient.

4. ICD-10-CM GUIDELINE COMPLIANCE
   The note language must not contradict standard ICD-10-CM coding guidelines.
   Examples of violations:
   - Documenting "acute on chronic" without selecting the appropriate combination code
   - Using "rule out" or "possible" in the assessment as if the condition is confirmed
   - Documenting an active condition as "personal history of"

5. CODE ASSIGNMENT ACCURACY
   Diagnoses stated in the note's Assessment/Plan must align with the required
   ICD-10 codes; no additional diagnoses that would require separate codes not
   in the required list, and no stated diagnoses that contradict a required code.

RESPONSE FORMAT
Return a JSON object with exactly two keys:
  "note_fix_needed": true if any compliance issue is found, false if fully compliant
  "fixes": array of specific, self-contained fix instructions for the note generator
           (empty array when note_fix_needed is false)

Each fix instruction must name the specific problem and state exactly what to change.
Example: "Remove the second mention of moderate persistent asthma in the Plan section;
J45.41 must appear exactly once in the coded encounter."
"""

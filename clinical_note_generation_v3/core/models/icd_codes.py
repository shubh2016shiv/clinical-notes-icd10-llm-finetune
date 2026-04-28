"""
ICD code data contracts for the v3 pipeline.

LAYER: core/models
ARCHITECTURE:
  Official ICD-10-CM order file records
      -> ICDCodeRecord (parsed row with computed properties)
      -> CandidateCode (retrieval result presented to selector)
      -> SelectedCode (selector-approved code with evidence)
      -> RejectedCandidate (selector-rejected code with reason)
      -> MissingCandidateCode (documented condition with no candidate)

DATA FLOW:
  Raw text line -> ICDCodeRecord -> CandidateCode -> SelectedCode | RejectedCandidate

DEPENDENCIES:
  - pydantic (BaseModel, computed_field, field_validator, Field)
  - typing (Literal)
"""

from typing import Literal

from pydantic import BaseModel, Field, computed_field, field_validator


class ICDCodeRecord(BaseModel):
    """
    Official ICD-10-CM code record parsed from the order file.

    Computed properties (normalized_code, chapter_prefix, search_document) are
    derived from the raw code field and used downstream for retrieval indexing.

    Args:
        code: ICD-10-CM code without dot formatting (e.g., "E119").
        is_billable: Whether this row is a billable diagnosis code.
        short_description: Official short description (≤ 60 characters).
        long_description: Official full description.
        order_number: Optional sequential order number from the official file.

    Returns:
        Validated ICD code record with computed fields.

    Raises:
        pydantic.ValidationError: If required fields are missing or have wrong types.

    Example:
        >>> record = ICDCodeRecord(code="E119", is_billable=True, short_description="T2DM", long_description="Type 2 diabetes mellitus without complications")
        >>> record.normalized_code
        'E11.9'
    """

    code: str
    is_billable: bool
    short_description: str
    long_description: str
    order_number: int | None = None

    @computed_field
    @property
    def normalized_code(self) -> str:
        """Return conventional dotted ICD-10-CM formatting when applicable."""
        if "." in self.code or len(self.code) <= 3:
            return self.code
        return f"{self.code[:3]}.{self.code[3:]}"

    @computed_field
    @property
    def chapter_prefix(self) -> str:
        """Return the leading ICD chapter letter (e.g., 'E' for endocrine)."""
        return self.code[:1]

    @computed_field
    @property
    def search_document(self) -> str:
        """Return the concatenated text indexed for vector and BM25 retrieval."""
        return f"{self.normalized_code} {self.long_description} {self.short_description}"


class CandidateCode(BaseModel):
    """
    Candidate ICD code provided to the constrained selector.

    Retrieved from FAISS vector index or BM25 lexical index. The candidate_id
    is assigned before the selector prompt so the model can reference codes by
    stable integer id rather than by raw string.

    Args:
        candidate_id: Stable one-based display id assigned before selection.
        code: Dotted ICD-10-CM code (e.g., "E11.9").
        description: Official long description.
        source: Retrieval source label ("faiss" or "bm25").
        score: Retrieval score (cosine similarity or BM25).
    """

    candidate_id: int | None = None
    code: str
    description: str
    source: str = Field(default="retrieval")
    score: float | None = None


class SelectedCode(BaseModel):
    """
    Selector-approved ICD code with evidence and rationale.

    Args:
        candidate_id: Matching candidate_id from the presented candidate list.
        code: Selected ICD-10-CM code.
        description: Official description confirming candidate alignment.
        evidence: Short quote or paraphrase from the clinical note.
        rationale: Brief clinical justification for selection.
    """

    candidate_id: int | None = None
    code: str
    description: str
    evidence: str
    rationale: str


RejectedReason = Literal[
    "negated",
    "uncertain",
    "historical_only",
    "unsupported",
    "less_specific",
    "not_relevant",
    "non_billable",
    "outside_candidates",
]


class RejectedCandidate(BaseModel):
    """
    Candidate ICD code explicitly rejected by the selector with a documented reason.

    Args:
        candidate_id: Matching candidate_id from the presented candidate list.
        code: Rejected ICD-10-CM code.
        reason: Canonical rejection reason from the RejectedReason literal set.

    Note:
        The field_validator normalizes free-form LLM reason strings (e.g.,
        "unsupported - no fracture documented") to the canonical literal value
        ("unsupported"), preventing ValidationError from model verbosity.
    """

    candidate_id: int | None = None
    code: str
    reason: RejectedReason

    @field_validator("reason", mode="before")
    @classmethod
    def normalize_rejection_reason(cls, value: object) -> object:
        """
        Normalize model-provided rejection reason strings to canonical literals.

        Args:
            value: Raw reason value from the selector JSON response.

        Returns:
            Canonical RejectedReason value when a matching prefix is found.

        Raises:
            None (unrecognized values pass through for Pydantic to handle).

        Example:
            >>> RejectedCandidate.normalize_rejection_reason("unsupported - no sprain documented")
            'unsupported'
        """
        if not isinstance(value, str):
            return value
        lowered_value = value.strip().lower()
        for allowed_reason in RejectedReason.__args__:
            if lowered_value == allowed_reason:
                return allowed_reason
            if lowered_value.startswith(f"{allowed_reason} "):
                return allowed_reason
            if lowered_value.startswith(f"{allowed_reason}-"):
                return allowed_reason
            if lowered_value.startswith(f"{allowed_reason}:"):
                return allowed_reason
        return lowered_value


class MissingCandidateCode(BaseModel):
    """
    Documented clinical condition for which no suitable candidate code was retrieved.

    Used by the selector to flag retrieval gaps before they become training noise.

    Args:
        condition: Human-readable condition name from the clinical note.
        reason: Explanation of why no correct candidate was available.
    """

    condition: str
    reason: str

"""
ICD condition-to-code resolver.

Takes a ClinicalBundleTemplate whose active_conditions are human-readable
strings and produces a SeededClinicalBundle where every condition has been
resolved to an official ICD-10-CM code.

Resolution strategy per condition
-----------------------------------
  1. Retrieve the top-N ICD candidate codes using mandatory hybrid BM25 + FAISS
     retrieval over the official ICD-10-CM description corpus.
  2. Send the candidates + the condition name + encounter context to the LLM
     via a focused single-condition selection prompt.
  3. Parse the LLM's JSON response and validate the returned code against
     the official repository.
  4. Return a ResolvedConditionWithIcdCode entry.

If any condition cannot be resolved, IcdResolutionFailedForConditionError is
raised so the pipeline orchestrator can decide whether to skip the bundle
template or surface the failure.

Typical usage
-------------
  resolver = IcdConditionToCodeResolver.from_default_settings()
  seeded_bundle = resolver.resolve_all_conditions_in_bundle_to_seeded_bundle(template)

Public classes
--------------
  IcdResolutionFailedForConditionError
  IcdConditionToCodeResolver
"""

from __future__ import annotations

import logging

from clinical_note_generation_v3.core.models.bundle import (
    ClinicalBundleTemplate,
    SeededClinicalBundle,
    ResolvedConditionCode,
)
from clinical_note_generation_v3.core.models.icd_codes import CandidateCode
from clinical_note_generation_v3.core.retrieval.hybrid_icd_candidate_retriever import (
    HybridICDCandidateRetriever,
)
from clinical_note_generation_v3.infrastructure.data_preprocessing.official_icd_loader import (
    OfficialICDCodeRepository,
)
from clinical_note_generation_v3.infrastructure.llm_provider.fallback_llm_client import (
    FallbackJSONClient,
    parse_json_from_provider_response,
)
from clinical_note_generation_v3.prompt_specs.registry import (
    build_icd_resolution_prompt_spec,
)
from clinical_note_generation_v3.prompt_specs.rendering import compose_chat_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class IcdResolutionFailedForConditionError(RuntimeError):
    """
    Raised when the ICD resolver cannot find a suitable code for a condition.

    This is a recoverable pipeline error — the orchestrator should catch it,
    log the failure, and either skip the bundle template or substitute a
    fallback template.

    Attributes
    ----------
    condition_name
        The condition string that failed to resolve.
    reason
        A short human-readable explanation of why resolution failed.
    """

    def __init__(self, *, condition_name: str, reason: str) -> None:
        self.condition_name = condition_name
        self.reason = reason
        super().__init__(f"ICD resolution failed for condition '{condition_name}': {reason}")


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


class IcdConditionToCodeResolver:
    """
    Resolves each human-readable active condition in a ClinicalBundleTemplate
    to an official ICD-10-CM code, returning a SeededClinicalBundle.

    Parameters
    ----------
    icd_official_repository
        The loaded OfficialICDCodeRepository used for post-selection code
        validation and for populating description fields on the result.
    hybrid_icd_candidate_retriever
        The retriever used to fetch BM25 (and optionally FAISS) candidates
        for each condition name.
    llm_json_client
        The LLM client used to select the best candidate code.
        Must implement generate_json(prompt) -> dict.
    candidate_count_per_condition
        How many retrieval candidates to fetch per condition and present to
        the LLM.  20–40 is a good range: enough coverage without overloading
        the prompt context.
    """

    def __init__(
        self,
        *,
        icd_official_repository: OfficialICDCodeRepository,
        hybrid_icd_candidate_retriever: HybridICDCandidateRetriever,
        llm_json_client: FallbackJSONClient,
        candidate_count_per_condition: int = 30,
    ) -> None:
        self._icd_repository = icd_official_repository
        self._hybrid_retriever = hybrid_icd_candidate_retriever
        self._llm_client = llm_json_client
        self._candidate_count_per_condition = candidate_count_per_condition

    # ------------------------------------------------------------------
    # Primary public method
    # ------------------------------------------------------------------

    def resolve_seeded_clinical_bundle_from_template(
        self,
        bundle_template: ClinicalBundleTemplate,
    ) -> SeededClinicalBundle:
        """
        Resolves every active condition in a bundle template to an ICD code.

        Returns a SeededClinicalBundle that is ready for constraint
        extraction and note generation.

        Raises
        ------
        IcdResolutionFailedForConditionError
            When any single condition cannot be resolved to a valid billable code.
            The caller should catch this and decide whether to retry with a
            different template or log the failure for analysis.
        """
        resolved_condition_entries: list[ResolvedConditionCode] = []

        for condition_name in bundle_template.active_conditions:
            logger.debug(
                "Resolving condition '%s' for template '%s'",
                condition_name,
                bundle_template.template_id,
            )
            resolved_condition_entry = self.resolve_condition_name_to_icd_code(
                condition_name=condition_name,
                bundle_archetype=bundle_template.archetype,
                encounter_context=bundle_template.encounter_context,
            )
            resolved_condition_entries.append(resolved_condition_entry)
            logger.debug(
                "  -> resolved '%s' to %s",
                condition_name,
                resolved_condition_entry.icd_code,
            )

        return SeededClinicalBundle(
            template_id=bundle_template.template_id,
            archetype=bundle_template.archetype,
            encounter_context=bundle_template.encounter_context,
            active_condition_names=list(bundle_template.active_conditions),
            resolved_conditions=resolved_condition_entries,
            allowed_distractors=list(bundle_template.allowed_distractors),
            trap_patterns=list(bundle_template.trap_patterns),
        )

    # ------------------------------------------------------------------
    # Single-condition resolution (can be called directly for debugging)
    # ------------------------------------------------------------------

    def resolve_condition_name_to_icd_code(
        self,
        *,
        condition_name: str,
        bundle_archetype: str,
        encounter_context: str,
    ) -> ResolvedConditionCode:
        """
        Resolves one condition name to a single official ICD-10-CM code.

        Steps:
          1. Retrieve BM25/hybrid candidates for this condition name.
          2. Build the single-condition selection prompt.
          3. Ask the LLM to select the best code from candidates.
          4. Validate the returned code against the official repository.
          5. Return a populated ResolvedConditionCode.

        Raises
        ------
        IcdResolutionFailedForConditionError
            When no candidates are found, the LLM declines to select a code,
            or the returned code is not a valid billable code in the official file.
        """
        retrieved_candidate_codes = self._retrieve_candidate_codes_for_condition_name(
            condition_name
        )

        if not retrieved_candidate_codes:
            raise IcdResolutionFailedForConditionError(
                condition_name=condition_name,
                reason=(
                    "BM25 retrieval returned zero candidates. "
                    "The condition name may be too unusual or abbreviated."
                ),
            )

        return self._select_best_icd_code_from_candidates(
            condition_name=condition_name,
            bundle_archetype=bundle_archetype,
            encounter_context=encounter_context,
            retrieved_candidate_codes=retrieved_candidate_codes,
        )

    # ------------------------------------------------------------------
    # Private: retrieval
    # ------------------------------------------------------------------

    def _retrieve_candidate_codes_for_condition_name(
        self,
        condition_name: str,
    ) -> list[CandidateCode]:
        """
        Uses the hybrid retriever to fetch candidate ICD codes for one condition.

        Passes the condition name as both the `clinical_note` text and the sole
        `focus_term`, which is the correct usage for single-condition lookup.
        """
        return self._hybrid_retriever.retrieve(
            clinical_note=condition_name,
            focus_terms=[condition_name],
            limit=self._candidate_count_per_condition,
        )

    # ------------------------------------------------------------------
    # Private: LLM-based code selection
    # ------------------------------------------------------------------

    def _select_best_icd_code_from_candidates(
        self,
        *,
        condition_name: str,
        bundle_archetype: str,
        encounter_context: str,
        retrieved_candidate_codes: list[CandidateCode],
    ) -> ResolvedConditionCode:
        """
        Sends the candidates and condition context to the LLM, parses the
        returned code, and validates it against the official repository.

        Returns a ResolvedConditionCode on success.

        Raises
        ------
        IcdResolutionFailedForConditionError
            When the LLM declines to select a code (resolution_succeeded=false),
            returns unparseable JSON, or returns a code that is not billable
            in the official ICD file.
        """
        prompt_spec = build_icd_resolution_prompt_spec(
            condition_name=condition_name,
            bundle_archetype=bundle_archetype,
            encounter_context=encounter_context,
            retrieved_candidates=retrieved_candidate_codes,
        )
        selection_prompt = compose_chat_prompt(
            system_prompt=prompt_spec.system_prompt,
            user_prompt=prompt_spec.user_prompt,
        )

        raw_llm_response_text = self._llm_client.generate_json(
            selection_prompt,
            response_schema=prompt_spec.response_schema,
        )

        # generate_json may return a dict directly (Gemini) or a raw string
        # (some fallback paths).  Normalise to dict.
        if isinstance(raw_llm_response_text, str):
            parsed_response = parse_json_from_provider_response(raw_llm_response_text)
        elif isinstance(raw_llm_response_text, dict):
            parsed_response = raw_llm_response_text
        else:
            raise IcdResolutionFailedForConditionError(
                condition_name=condition_name,
                reason=(
                    f"LLM client returned an unexpected type "
                    f"({type(raw_llm_response_text).__name__}) instead of dict or str."
                ),
            )

        return self._validate_and_build_resolved_condition_code(
            condition_name=condition_name,
            parsed_llm_response=parsed_response,
            retrieved_candidate_codes=retrieved_candidate_codes,
            prompt_id=prompt_spec.prompt_id,
            prompt_version=prompt_spec.prompt_version,
        )

    def _validate_and_build_resolved_condition_code(
        self,
        *,
        condition_name: str,
        parsed_llm_response: dict,
        retrieved_candidate_codes: list[CandidateCode],
        prompt_id: str,
        prompt_version: str,
    ) -> ResolvedConditionCode:
        """
        Validates the LLM's code selection and builds the resolved condition entry.

        Validation steps:
          1. Check resolution_succeeded flag.
          2. Extract and normalise the returned code string.
          3. Look the code up in the official repository.
          4. Confirm it is billable (not a header code).
          5. Confirm it was in the retrieved candidate list.

        Step 5 is intentional: accepting any valid billable code regardless of
        whether it was retrieved would allow the LLM to hallucinate a real-but-wrong
        code that the rest of the pipeline would faithfully treat as ground truth.
        Constraining to the retrieval set preserves upstream label control.
        """
        resolution_succeeded = parsed_llm_response.get("resolution_succeeded", False)

        if not resolution_succeeded:
            rationale = parsed_llm_response.get("selection_rationale", "no rationale provided")
            raise IcdResolutionFailedForConditionError(
                condition_name=condition_name,
                reason=f"LLM declined to select a code: {rationale}",
            )

        returned_code_string = parsed_llm_response.get("selected_icd_code")
        if not returned_code_string or not isinstance(returned_code_string, str):
            raise IcdResolutionFailedForConditionError(
                condition_name=condition_name,
                reason=(
                    "LLM returned resolution_succeeded=true but selected_icd_code "
                    "is missing or not a string."
                ),
            )

        normalised_code_for_lookup = returned_code_string.strip()
        official_icd_record = self._icd_repository.get_code(normalised_code_for_lookup)

        if official_icd_record is None:
            raise IcdResolutionFailedForConditionError(
                condition_name=condition_name,
                reason=(
                    f"LLM returned code '{returned_code_string}' which does not exist "
                    f"in the official ICD-10-CM repository. "
                    f"The model may have hallucinated or recalled an outdated code."
                ),
            )

        if not official_icd_record.is_billable:
            raise IcdResolutionFailedForConditionError(
                condition_name=condition_name,
                reason=(
                    f"LLM returned code '{returned_code_string}' "
                    f"({official_icd_record.short_description}) which is a "
                    f"NON-BILLABLE header code. A more specific billable subcode is required."
                ),
            )

        candidate_code_set = {c.code.strip().upper() for c in retrieved_candidate_codes}
        if official_icd_record.normalized_code.upper() not in candidate_code_set:
            raise IcdResolutionFailedForConditionError(
                condition_name=condition_name,
                reason=(
                    f"LLM returned code '{returned_code_string}' "
                    f"({official_icd_record.short_description}) which is valid and billable "
                    f"but was not in the {len(retrieved_candidate_codes)} retrieved candidates. "
                    f"Only codes from the retrieval set are accepted to maintain upstream "
                    f"label control. Increase candidate_count_per_condition if this condition "
                    f"consistently fails to retrieve the correct code."
                ),
            )

        return ResolvedConditionCode(
            condition_name=condition_name,
            icd_code=official_icd_record.normalized_code,
            icd_short_description=official_icd_record.short_description,
            icd_long_description=official_icd_record.long_description,
            resolver_prompt_id=prompt_id,
            resolver_prompt_version=prompt_version,
        )

    # ------------------------------------------------------------------
    # Convenience factory — wires the resolver from environment settings
    # ------------------------------------------------------------------

    @classmethod
    def from_default_settings(
        cls,
        *,
        candidate_count_per_condition: int = 30,
    ) -> "IcdConditionToCodeResolver":
        """
        Creates a fully wired resolver using the default V3 pipeline settings.

        Uses mandatory hybrid BM25 + FAISS retrieval with a DeepSeek primary LLM
        and OpenAI fallback.  The FAISS manifest controls which embedding
        provider/model is used at retrieval time.

        Parameters
        ----------
        candidate_count_per_condition
            Number of candidates to retrieve and present to the LLM per condition.
        """
        from clinical_note_generation_v3.config.settings import V3PipelineSettings
        from clinical_note_generation_v3.infrastructure.embedding_provider.embedding_client_factory import (
            create_embedding_client,
        )
        from clinical_note_generation_v3.infrastructure.llm_provider.llm_client_factory import (
            create_default_json_generation_client,
        )
        from clinical_note_generation_v3.infrastructure.vector_store.faiss_icd_candidate_index import (
            FAISSICDCandidateIndex,
        )

        settings = V3PipelineSettings()

        icd_repository = OfficialICDCodeRepository(settings.official_icd_order_path)

        manifest = FAISSICDCandidateIndex.read_manifest(settings.faiss_persist_directory)
        if str(manifest.get("embedding_provider", "")).lower() != "openai":
            raise RuntimeError(
                f"{FAISSICDCandidateIndex.MANDATORY_HYBRID_REMEDIATION} "
                f"Default v3 retrieval requires an OpenAI FAISS manifest, got "
                f"{manifest.get('embedding_provider')}/{manifest.get('embedding_model')}."
            )
        embedding_client = create_embedding_client(
            provider_name=str(manifest["embedding_provider"]),
            model_name=str(manifest["embedding_model"]),
            settings=settings,
        )
        vector_index = FAISSICDCandidateIndex(
            repository=icd_repository,
            embedding_client=embedding_client,
            persist_directory=settings.faiss_persist_directory,
            prefer_gpu=settings.prefer_faiss_gpu,
        )
        if vector_index.count <= 0:
            raise RuntimeError(FAISSICDCandidateIndex.MANDATORY_HYBRID_REMEDIATION)
        logger.info(
            "Loaded mandatory hybrid ICD vector index from %s using backend=%s count=%d embedding=%s/%s",
            settings.faiss_persist_directory,
            vector_index.backend_name,
            vector_index.count,
            manifest["embedding_provider"],
            manifest["embedding_model"],
        )

        hybrid_retriever = HybridICDCandidateRetriever(
            repository=icd_repository,
            vector_index=vector_index,
        )

        return cls(
            icd_official_repository=icd_repository,
            hybrid_icd_candidate_retriever=hybrid_retriever,
            llm_json_client=create_default_json_generation_client(settings),
            candidate_count_per_condition=candidate_count_per_condition,
        )

    # ------------------------------------------------------------------
    # Backward-compatible wrappers during naming transition
    # ------------------------------------------------------------------

    def resolve_all_conditions_in_bundle_to_seeded_bundle(
        self,
        bundle_template: ClinicalBundleTemplate,
    ) -> SeededClinicalBundle:
        return self.resolve_seeded_clinical_bundle_from_template(bundle_template)

    def resolve_single_condition_name_to_icd_code(
        self,
        *,
        condition_name: str,
        bundle_archetype: str,
        encounter_context: str,
    ) -> ResolvedConditionCode:
        return self.resolve_condition_name_to_icd_code(
            condition_name=condition_name,
            bundle_archetype=bundle_archetype,
            encounter_context=encounter_context,
        )

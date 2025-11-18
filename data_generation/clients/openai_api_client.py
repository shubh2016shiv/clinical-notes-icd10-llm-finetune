"""
OpenAI API Client

This module provides a client for interacting with OpenAI's GPT models to generate
clinical notes and validate ICD-10 code assignments.

Author: RhythmX AI Team
Date: November 2025
"""

import json
from typing import Dict, List

from loguru import logger
from openai import OpenAI

from ..core.data_models import ICD10Code
from ..core.enums import ClinicalNoteType
from ..prompts import get_clinical_note_generation_prompt, get_icd10_validation_prompt


class OpenAIAPIClient:
    """
    Handles all interactions with OpenAI's GPT models for clinical note generation.

    This class manages the OpenAI API client configuration, clinical note generation,
    and ICD-10 code validation using AI-powered analysis. It includes error handling,
    rate limiting, and structured output parsing.

    Attributes:
        api_key: OpenAI API key
        model_name: Name of the OpenAI model to use
        client: Initialized OpenAI client instance
        generation_config: Configuration dictionary for text generation

    Example:
        >>> client = OpenAIAPIClient(
        ...     api_key="your_api_key",
        ...     model_name="gpt-4o"
        ... )
        >>> note = client.generate_clinical_note(
        ...     note_type=ClinicalNoteType.PROGRESS_NOTE,
        ...     target_icd10_codes=[code1, code2]
        ... )
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o",
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 8192
    ):
        """
        Initialize OpenAI API client with configuration.

        Step 1: Store API configuration
        Step 2: Initialize OpenAI client
        Step 3: Store generation configuration
        Step 4: Log initialization success

        Args:
            api_key: OpenAI API key
            model_name: Name of the OpenAI model to use (default: gpt-4o)
            temperature: Sampling temperature (0.0-1.0, higher = more creative)
            top_p: Nucleus sampling parameter (0.0-1.0)
            max_tokens: Maximum tokens in generated output

        Example:
            >>> client = OpenAIAPIClient(api_key="your_key")
        """
        # Step 1: Store API configuration
        self.api_key = api_key
        self.model_name = model_name

        # Step 2: Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

        # Step 3: Store generation configuration
        self.generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        # Step 4: Log initialization success
        logger.info("=" * 80)
        logger.info("OPENAI API CLIENT INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"Model Name: {self.model_name}")
        logger.info(f"Temperature: {temperature}")
        logger.info(f"Top-P: {top_p}")
        logger.info(f"Max Tokens: {max_tokens}")
        logger.info("=" * 80)

    def generate_clinical_note(
        self,
        note_type: ClinicalNoteType,
        target_icd10_codes: List[ICD10Code],
        additional_context: str = ""
    ) -> str:
        """
        Generate a realistic clinical note using OpenAI GPT model.

        Step 1: Build ICD-10 codes description for prompt
        Step 2: Construct comprehensive generation prompt
        Step 3: Call OpenAI API to generate note
        Step 4: Extract and clean generated text
        Step 5: Return generated note

        Args:
            note_type: Type of clinical note to generate
            target_icd10_codes: List of ICD-10 codes to incorporate in note
            additional_context: Additional context for note generation (optional)

        Returns:
            Generated clinical note as string

        Raises:
            Exception: If API call fails or returns invalid response

        Example:
            >>> note = client.generate_clinical_note(
            ...     note_type=ClinicalNoteType.PROGRESS_NOTE,
            ...     target_icd10_codes=[code1, code2]
            ... )
        """
        logger.debug(f"Generating {note_type.value} with {len(target_icd10_codes)} ICD-10 codes")

        # Step 1 & 2: Get comprehensive generation prompt from centralized prompt module
        prompt = get_clinical_note_generation_prompt(
            note_type=note_type,
            target_icd10_codes=target_icd10_codes,
            additional_context=additional_context
        )

        try:
            # Step 3: Call OpenAI API to generate note
            logger.debug("Calling OpenAI API for clinical note generation...")

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a medical documentation specialist."},
                    {"role": "user", "content": prompt}
                ],
                **self.generation_config
            )

            # Step 4: Extract and clean generated text
            generated_note = response.choices[0].message.content.strip()

            logger.debug(
                f"Successfully generated clinical note "
                f"({len(generated_note)} characters)"
            )

            # Step 5: Return generated note
            return generated_note

        except Exception as error:
            error_message = f"Error generating clinical note: {error}"
            logger.error(error_message)
            raise Exception(error_message)

    def validate_icd10_assignment(
        self,
        clinical_note: str,
        assigned_icd10_codes: List[str],
        icd10_reference_data: List[ICD10Code]
    ) -> Dict:
        """
        Use OpenAI to validate if assigned ICD-10 codes match the clinical note.

        Step 1: Build codes description for validation prompt
        Step 2: Construct validation prompt
        Step 3: Call OpenAI API for validation
        Step 4: Parse JSON response from validation
        Step 5: Return validation results

        Args:
            clinical_note: The clinical note text to validate
            assigned_icd10_codes: List of ICD-10 code strings assigned to note
            icd10_reference_data: Reference data for the assigned codes

        Returns:
            Dictionary with validation results including:
                - valid_codes: List of valid ICD-10 codes
                - invalid_codes: List of invalid codes with reasons
                - missing_codes: List of diagnoses in note but not coded
                - confidence_score: Numeric score 0-100
                - validation_notes: Human-readable explanation

        Example:
            >>> result = client.validate_icd10_assignment(
            ...     clinical_note="Patient presents with...",
            ...     assigned_icd10_codes=["E11.9", "I10"],
            ...     icd10_reference_data=[code1, code2]
            ... )
        """
        logger.debug(f"Validating {len(assigned_icd10_codes)} ICD-10 code assignments")

        # Step 1 & 2: Get comprehensive validation prompt from centralized prompt module
        prompt = get_icd10_validation_prompt(
            clinical_note=clinical_note,
            icd10_reference_data=icd10_reference_data
        )

        try:
            # Step 3: Call OpenAI API for validation
            logger.debug("Calling OpenAI API for ICD-10 validation...")

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a medical coding auditor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Lower temperature for validation
                max_tokens=2048
            )

            # Step 4: Parse JSON response from validation
            validation_text = response.choices[0].message.content.strip()

            # Handle potential markdown formatting
            if validation_text.startswith("```json"):
                validation_text = validation_text.replace("```json", "").replace("```", "").strip()
            elif validation_text.startswith("```"):
                validation_text = validation_text.replace("```", "").strip()

            validation_result = json.loads(validation_text)

            logger.debug(
                f"Validation complete - Confidence: "
                f"{validation_result.get('confidence_score', 0)}%"
            )

            # Step 5: Return validation results
            return validation_result

        except json.JSONDecodeError as error:
            logger.error(f"Failed to parse validation JSON response: {error}")
            logger.debug(f"Raw response: {validation_text}")

            # Return default validation result in case of parsing error
            return {
                "valid_codes": assigned_icd10_codes,
                "invalid_codes": [],
                "missing_codes": [],
                "confidence_score": 0,
                "validation_notes": f"Validation parsing error: {str(error)}"
            }

        except Exception as error:
            error_message = f"Error validating ICD-10 codes: {error}"
            logger.error(error_message)

            # Return default validation result in case of error
            return {
                "valid_codes": assigned_icd10_codes,
                "invalid_codes": [],
                "missing_codes": [],
                "confidence_score": 0,
                "validation_notes": f"Validation error: {str(error)}"
            }


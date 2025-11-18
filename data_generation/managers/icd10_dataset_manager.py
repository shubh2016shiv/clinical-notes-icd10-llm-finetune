"""
ICD-10 Dataset Manager

This module provides functionality for loading, querying, and managing the ICD-10
diagnosis code dataset used for clinical note generation.

Author: RhythmX AI Team
Date: November 2025
"""

import json
import random
from typing import Dict, List, Optional

from loguru import logger

from ..core.data_models import ICD10Code


class ICD10DatasetManager:
    """
    Manages loading and querying of ICD-10 codes from the reference dataset.
    
    This class handles all operations related to the ICD-10 code dataset including
    loading from JSON, indexing for fast lookups, and providing query methods for
    clinical note generation. It maintains multiple data structures for efficient
    access patterns.
    
    Attributes:
        dataset_file_path: Path to the JSON file containing ICD-10 codes
        icd10_codes_dictionary: Dictionary mapping code IDs to ICD10Code objects
        billable_codes_list: List of all billable ICD-10 codes
        diagnosis_categories_map: Dictionary mapping categories to lists of codes
    
    Example:
        >>> manager = ICD10DatasetManager("data_generation/icd10_dataset.json")
        >>> code = manager.get_code_by_id("E11.9")
        >>> print(code.icd10_name)
        'Type 2 diabetes mellitus without complications'
    """
    
    def __init__(self, dataset_file_path: str):
        """
        Initialize the ICD-10 dataset manager.
        
        Step 1: Store dataset file path
        Step 2: Initialize data structures
        Step 3: Load and parse ICD-10 dataset
        
        Args:
            dataset_file_path: Path to the JSON file containing ICD-10 codes
            
        Raises:
            FileNotFoundError: If dataset file is not found
            ValueError: If dataset file contains invalid JSON
            
        Example:
            >>> manager = ICD10DatasetManager("data_generation/icd10_dataset.json")
        """
        # Step 1: Store dataset file path
        self.dataset_file_path = dataset_file_path
        
        # Step 2: Initialize data structures
        self.icd10_codes_dictionary: Dict[str, ICD10Code] = {}
        self.billable_codes_list: List[ICD10Code] = []
        self.diagnosis_categories_map: Dict[str, List[ICD10Code]] = {}
        
        # Step 3: Load and parse ICD-10 dataset
        self._load_icd10_dataset()
    
    def _load_icd10_dataset(self) -> None:
        """
        Load and parse the ICD-10 dataset from JSON file.
        
        Step 1: Log dataset loading start
        Step 2: Open and read JSON file
        Step 3: Parse each record into ICD10Code object
        Step 4: Build data structures for efficient querying
        Step 4.1: Store in main dictionary by code ID
        Step 4.2: Add billable codes to separate list
        Step 4.3: Group codes by diagnosis category
        Step 5: Log dataset loading statistics
        
        Raises:
            FileNotFoundError: If dataset file is not found at specified path
            json.JSONDecodeError: If JSON file is malformed
            KeyError: If required fields are missing from records
        """
        # Step 1: Log dataset loading start
        logger.info("=" * 80)
        logger.info("LOADING ICD-10 DATASET")
        logger.info("=" * 80)
        logger.info(f"Dataset file path: {self.dataset_file_path}")
        
        try:
            # Step 2: Open and read JSON file
            with open(self.dataset_file_path, 'r', encoding='utf-8') as file:
                raw_data = json.load(file)
            
            logger.info(f"Total records found in dataset: {len(raw_data):,}")
            
            # Step 3: Parse each record into ICD10Code object
            for record in raw_data:
                icd10_code_object = ICD10Code(
                    icd10_code=record.get('icd10_code', ''),
                    icd10_name=record.get('icd10_name', ''),
                    is_billable=record.get('is_billable', False),
                    diagnosis_category=record.get('diagnosis_category', ''),
                    category_code=record.get('category_code', ''),
                    category_description=record.get('category_description', ''),
                    hcc_raf=record.get('hcc_raf')
                )
                
                # Step 4.1: Store in main dictionary by code ID
                self.icd10_codes_dictionary[icd10_code_object.icd10_code] = icd10_code_object
                
                # Step 4.2: Add billable codes to separate list
                if icd10_code_object.is_billable:
                    self.billable_codes_list.append(icd10_code_object)
                
                # Step 4.3: Group codes by diagnosis category
                category = icd10_code_object.diagnosis_category
                if category not in self.diagnosis_categories_map:
                    self.diagnosis_categories_map[category] = []
                self.diagnosis_categories_map[category].append(icd10_code_object)
            
            # Step 5: Log dataset loading statistics
            logger.info("=" * 80)
            logger.info("ICD-10 DATASET LOADED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info(f"Total ICD-10 codes loaded: {len(self.icd10_codes_dictionary):,}")
            logger.info(f"Billable codes available: {len(self.billable_codes_list):,}")
            logger.info(f"Diagnosis categories: {len(self.diagnosis_categories_map):,}")
            logger.info("=" * 80)
            
        except FileNotFoundError:
            error_message = f"ICD-10 dataset file not found: {self.dataset_file_path}"
            logger.error(error_message)
            raise FileNotFoundError(error_message)
        
        except json.JSONDecodeError as error:
            error_message = f"Invalid JSON in ICD-10 dataset file: {error}"
            logger.error(error_message)
            raise ValueError(error_message)
        
        except Exception as error:
            error_message = f"Unexpected error loading ICD-10 dataset: {error}"
            logger.error(error_message)
            raise
    
    def get_code_by_id(self, icd10_code: str) -> Optional[ICD10Code]:
        """
        Retrieve ICD-10 code details by code ID.
        
        Step 1: Look up code in dictionary
        Step 2: Return code object if found, None otherwise
        
        Args:
            icd10_code: The ICD-10 code to look up (e.g., "E11.9")
            
        Returns:
            ICD10Code object if found, None otherwise
            
        Example:
            >>> manager = ICD10DatasetManager("data_generation/icd10_dataset.json")
            >>> code = manager.get_code_by_id("E11.9")
            >>> if code:
            ...     print(code.icd10_name)
        """
        # Step 1: Look up code in dictionary
        return self.icd10_codes_dictionary.get(icd10_code)
    
    def get_random_billable_codes(
        self,
        count: int,
        category: Optional[str] = None
    ) -> List[ICD10Code]:
        """
        Get random billable ICD-10 codes for clinical note generation.
        
        Step 1: Determine available codes pool
        Step 1.1: Filter by category if specified
        Step 1.2: Use all billable codes if no category specified
        Step 2: Handle edge cases
        Step 2.1: Return empty list if no codes available
        Step 2.2: Adjust sample size if larger than available codes
        Step 3: Randomly sample codes
        Step 4: Log selection
        
        Args:
            count: Number of codes to retrieve
            category: Optional diagnosis category to filter by
            
        Returns:
            List of random billable ICD10Code objects
            
        Example:
            >>> manager = ICD10DatasetManager("data_generation/icd10_dataset.json")
            >>> codes = manager.get_random_billable_codes(count=3)
            >>> print(len(codes))
            3
        """
        # Step 1.1: Filter by category if specified
        if category and category in self.diagnosis_categories_map:
            available_codes = [
                code for code in self.diagnosis_categories_map[category]
                if code.is_billable
            ]
            logger.debug(
                f"Filtered to category '{category}': "
                f"{len(available_codes)} billable codes available"
            )
        else:
            # Step 1.2: Use all billable codes
            available_codes = self.billable_codes_list
            logger.debug(f"Using all billable codes: {len(available_codes)} available")
        
        # Step 2.1: Return empty list if no codes available
        if not available_codes:
            logger.warning(f"No billable codes available for category: {category}")
            return []
        
        # Step 2.2: Adjust sample size if larger than available codes
        sample_size = min(count, len(available_codes))
        if sample_size < count:
            logger.warning(
                f"Requested {count} codes but only {sample_size} available. "
                f"Returning {sample_size} codes."
            )
        
        # Step 3: Randomly sample codes
        selected_codes = random.sample(available_codes, sample_size)
        
        # Step 4: Log selection
        logger.debug(
            f"Selected {len(selected_codes)} random billable ICD-10 codes: "
            f"{[code.icd10_code for code in selected_codes]}"
        )
        
        return selected_codes
    
    def validate_icd10_code(self, icd10_code: str) -> bool:
        """
        Validate if an ICD-10 code exists in the dataset.
        
        Step 1: Check if code exists in dictionary
        Step 2: Return boolean result
        
        Args:
            icd10_code: The code to validate
            
        Returns:
            True if code exists in dataset, False otherwise
            
        Example:
            >>> manager = ICD10DatasetManager("data_generation/icd10_dataset.json")
            >>> is_valid = manager.validate_icd10_code("E11.9")
            >>> print(is_valid)
            True
        """
        # Step 1: Check if code exists in dictionary
        is_valid = icd10_code in self.icd10_codes_dictionary
        
        # Step 2: Log validation result
        if not is_valid:
            logger.debug(f"ICD-10 code '{icd10_code}' not found in dataset")
        
        return is_valid
    
    def get_all_diagnosis_categories(self) -> List[str]:
        """
        Get list of all available diagnosis categories in the dataset.
        
        Step 1: Extract category names from map
        Step 2: Return as sorted list
        
        Returns:
            Sorted list of diagnosis category names
            
        Example:
            >>> manager = ICD10DatasetManager("data_generation/icd10_dataset.json")
            >>> categories = manager.get_all_diagnosis_categories()
            >>> print(categories[0])
            'Certain infectious and parasitic diseases'
        """
        # Step 1: Extract category names
        categories = list(self.diagnosis_categories_map.keys())
        
        # Step 2: Return sorted list
        return sorted(categories)


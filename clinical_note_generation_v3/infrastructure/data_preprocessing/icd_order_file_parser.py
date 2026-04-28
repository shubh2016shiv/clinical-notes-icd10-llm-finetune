"""
Official ICD-10-CM order-file parser.

LAYER: infrastructure/data_preprocessing
ARCHITECTURE:
  icd10cm-order-April-1-2026.txt  (fixed-width text file)
      -> parse_icd_order_line()   (single line -> ICDCodeRecord)
      -> iter_icd_order_records() (file -> Iterator[ICDCodeRecord])
      -> OfficialICDCodeRepository (in official_icd_loader.py)

  Fixed-width column layout (0-indexed, per WHO April 1 2026 release):
    [0:5]   order_number  (right-justified integer)
    [6:13]  code          (ICD code without dot)
    [14:15] billable_flag ("1" = billable, "0" = header)
    [16:76] short_description
    [77:]   long_description

DATA FLOW:
  str (raw line) -> parse_icd_order_line() -> ICDCodeRecord
  Path           -> iter_icd_order_records() -> Iterator[ICDCodeRecord]

DEPENDENCIES:
  - core/models/icd_codes.py (ICDCodeRecord)
  - stdlib (pathlib, collections.abc)
"""

from collections.abc import Iterator
from pathlib import Path

from clinical_note_generation_v3.core.models.icd_codes import ICDCodeRecord


def parse_icd_order_line(line: str) -> ICDCodeRecord:
    """
    Parse one fixed-width ICD-10-CM order-file line into an ICDCodeRecord.

    Args:
        line: Raw line from the official order file (may include trailing newline).

    Returns:
        Parsed ICDCodeRecord with all fields populated.

    Raises:
        ValueError: If the row is shorter than the minimum 17 characters required
                    to extract all fixed-width fields.
        ValueError: If the billable flag is not "0" or "1".

    Example:
        >>> parse_icd_order_line("00004 A009    1 Cholera, unspecified                                         Cholera, unspecified").normalized_code
        'A00.9'
    """
    raw_line = line.rstrip("\n")
    if len(raw_line) < 17:
        raise ValueError(f"ICD order row is too short to parse: {raw_line!r}")

    billable_flag = raw_line[14:15]
    if billable_flag not in {"0", "1"}:
        raise ValueError(f"Invalid billable flag {billable_flag!r} in row: {raw_line!r}")

    order_number_text = raw_line[0:5].strip()
    order_number = int(order_number_text) if order_number_text else None

    return ICDCodeRecord(
        order_number=order_number,
        code=raw_line[6:13].strip(),
        is_billable=billable_flag == "1",
        short_description=raw_line[16:76].strip(),
        long_description=raw_line[77:].strip() or raw_line[16:].strip(),
    )


def iter_icd_order_records(order_file_path: Path) -> Iterator[ICDCodeRecord]:
    """
    Yield ICDCodeRecord objects parsed from an official ICD-10-CM order file.

    Skips empty lines. Raises ValueError with the offending line number on any
    malformed row to make debugging the order file layout straightforward.

    Args:
        order_file_path: Absolute path to the official ICD-10-CM order file.

    Returns:
        Iterator yielding one ICDCodeRecord per non-empty line.

    Raises:
        FileNotFoundError: If the order file does not exist at the given path.
        ValueError: If any non-empty row is malformed (includes line number).

    Example:
        >>> from itertools import islice
        >>> path = Path("official_icd10cm_2026_april_1/icd10cm-order-April-1-2026.txt")
        >>> len(list(islice(iter_icd_order_records(path), 1))) in {0, 1}
        True
    """
    if not order_file_path.exists():
        raise FileNotFoundError(f"Official ICD-10-CM order file not found: {order_file_path}")

    with order_file_path.open("r", encoding="utf-8") as order_file:
        for line_number, line in enumerate(order_file, start=1):
            if not line.strip():
                continue
            try:
                yield parse_icd_order_line(line)
            except ValueError as parse_error:
                raise ValueError(
                    f"Failed to parse ICD order row at line {line_number}: {parse_error}"
                ) from parse_error

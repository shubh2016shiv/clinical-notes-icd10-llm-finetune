"""
Console progress reporter for human-readable per-note pipeline stages.
"""

from __future__ import annotations


class ConsoleProgressReporter:
    """
    Prints clear per-note stage progress for the v3 clinical note pipeline.
    """

    _HEADER_WIDTH = 101
    _STAGE_TEXT_WIDTH = 97
    _has_printed_any_note_header = False

    def __init__(
        self,
        *,
        note_number: int,
        total_count: int,
        correlation_id: str,
    ) -> None:
        self._note_number = note_number
        self._total_count = total_count
        self._correlation_id = correlation_id

    @property
    def correlation_id(self) -> str:
        return self._correlation_id

    def begin_note(self) -> None:
        if self.__class__._has_printed_any_note_header:
            print()
        print("*" * 50)
        print(
            f">> Clinical Note Generation {self._note_number}/{self._total_count} "
            f"[{self._correlation_id}]"
        )
        self.__class__._has_printed_any_note_header = True

    def stage(self, stage_number: int, description: str, model_label: str | None = None) -> None:
        stage_text = f"STAGE {stage_number}: {description}"
        if model_label:
            stage_text += f" | Model Used: {model_label}"

        print("-" * self._HEADER_WIDTH)
        print(f"| {stage_text.ljust(self._STAGE_TEXT_WIDTH)} |")
        print("-" * self._HEADER_WIDTH)

    def detail(self, message: str) -> None:
        print(f"  - {message}")

    def final_decision(self, decision_label: str, message: str) -> None:
        print(f"  - Final Decision: {decision_label}")
        if message:
            print(f"  - {message}")

    def end_note(self) -> None:
        print()

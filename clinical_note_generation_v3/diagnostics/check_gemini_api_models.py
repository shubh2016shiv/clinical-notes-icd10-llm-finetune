"""
Gemini API model availability diagnostic.

LAYER: diagnostics
ARCHITECTURE:
  GEMINI_API_KEY (via config/settings)
      -> Gemini models list endpoint
      -> per-model generateContent / embedContent ping
      -> ANSI-colored terminal table + working_gemini_models.md

  This script reads from infrastructure (config) and writes to stdout/filesystem.
  It does not import from core/ or application/ per the diagnostics layer rule.

DATA FLOW:
  API key -> model list -> per-model test -> grouped results -> table + markdown

DEPENDENCIES:
  - config/settings (read_secret_from_environment)
  - requests (external)
  - stdlib (os, time)
"""

import os
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clinical_note_generation_v3.config.settings import read_secret_from_environment  # noqa: E402

os.system("")  # Enable ANSI escape sequences on Windows cmd / PowerShell

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
GRAY = "\033[90m"


def test_gemini_model_endpoint(
    api_key: str,
    model_name: str,
    method: str,
) -> tuple[bool, str]:
    """
    Ping one Gemini model endpoint and return (success, detail) tuple.

    Args:
        api_key: Gemini API key.
        model_name: Full model name (e.g., "models/gemini-2.5-flash-preview-04-17").
        method: Gemini method to test ("generateContent" or "embedContent").

    Returns:
        Tuple of (worked: bool, detail_message: str).

    Raises:
        None.
    """
    base_url = (
        f"https://generativelanguage.googleapis.com/v1beta/{model_name}:{method}?key={api_key}"
    )
    if method == "generateContent":
        payload = {"contents": [{"parts": [{"text": "Hi"}]}]}
    elif method == "embedContent":
        payload = {"model": model_name, "content": {"parts": [{"text": "Hi"}]}}
    else:
        return False, "Unknown method"

    try:
        response = requests.post(base_url, json=payload)
        if response.status_code == 200:
            return True, "OK"
        error_message = response.json().get("error", {}).get("message", response.text)
        return False, _sanitize_api_key_from_message(
            f"HTTP {response.status_code}: {error_message}", api_key
        )
    except Exception as request_error:
        return False, _sanitize_api_key_from_message(str(request_error), api_key)


def _sanitize_api_key_from_message(message: str, api_key: str) -> str:
    if not message:
        return message
    return message.replace(api_key, "[REDACTED_API_KEY]")


def _format_result_row(
    model_name: str,
    works: bool,
    details: str,
    name_width: int,
    details_width: int,
) -> str:
    status_icon = "✔ OK" if works else "✘ ERR"
    status_color = GREEN if works else RED
    detail_text = (details if not works else "-").replace("\n", " ")
    if len(detail_text) > details_width:
        detail_text = detail_text[: details_width - 3] + "..."
    return (
        f"  {model_name.ljust(name_width)} | "
        f"{status_color}{status_icon.ljust(6)}{RESET} | "
        f"{GRAY}{detail_text.ljust(details_width)}{RESET}"
    )


def main() -> None:
    """Fetch available Gemini models and ping each endpoint, printing a formatted table."""
    api_key = read_secret_from_environment("GEMINI_API_KEY")
    if not api_key:
        print(f"{RED}Error: GEMINI_API_KEY not found in environment or .env file.{RESET}")
        return

    models_list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        print(f"{CYAN}Fetching available models from Gemini API...{RESET}")
        response = requests.get(models_list_url)
        response.raise_for_status()
        available_models = response.json().get("models", [])
    except Exception as fetch_error:
        print(
            f"{RED}Failed to fetch models list: {_sanitize_api_key_from_message(str(fetch_error), api_key)}{RESET}"
        )
        return

    results: list[dict] = []
    print(f"\n{BOLD}Initializing Model Evaluation... Please wait as we ping each API.{RESET}\n")

    for model in available_models:
        model_name = model["name"]
        supported_methods = model.get("supportedGenerationMethods", [])

        if "generateContent" in supported_methods:
            worked, reason = test_gemini_model_endpoint(api_key, model_name, "generateContent")
            results.append(
                {"Model": model_name, "Type": "Chat Completion", "Works": worked, "Details": reason}
            )
            time.sleep(0.5)

        if "embedContent" in supported_methods:
            worked, reason = test_gemini_model_endpoint(api_key, model_name, "embedContent")
            results.append(
                {"Model": model_name, "Type": "Embedding", "Works": worked, "Details": reason}
            )
            time.sleep(0.5)

    if not results:
        print(f"{RED}No models found.{RESET}")
        return

    grouped: dict[str, list[dict]] = {}
    for result in results:
        grouped.setdefault(result["Type"], []).append(result)

    name_width, details_width = 46, 50
    header = f"  {'Model Name'.ljust(name_width)} | {'Status'.ljust(6)} | {'Details'.ljust(details_width)}"
    separator = "-" * (name_width + 6 + details_width + 10)

    for group_type, items in grouped.items():
        items.sort(key=lambda item: not item["Works"])
        icon = "[MSG]" if group_type == "Chat Completion" else "[EMB]"
        print(f"{BOLD}{YELLOW}▶ {icon} {group_type.upper()} MODELS{RESET}")
        print(separator)
        print(f"{BOLD}{header}{RESET}")
        print(separator)
        for result in items:
            print(
                _format_result_row(
                    result["Model"], result["Works"], result["Details"], name_width, details_width
                )
            )
        print(separator + "\n")

    with open("working_gemini_models.md", "w", encoding="utf-8") as markdown_file:
        markdown_file.write("# Gemini Models Test Results\n\n")
        for group_type, items in grouped.items():
            markdown_file.write(
                f"## {group_type}\n\n| Model Name | Status | Details |\n|---|---|---|\n"
            )
            for result in items:
                status = "✅ Yes" if result["Works"] else "❌ No"
                detail = result["Details"].replace("\n", " ") if not result["Works"] else "-"
                markdown_file.write(f"| {result['Model']} | {status} | {detail} |\n")
            markdown_file.write("\n")

    print(f"{GREEN}{BOLD}Evaluation complete. Markdown saved to 'working_gemini_models.md'{RESET}")


if __name__ == "__main__":
    main()

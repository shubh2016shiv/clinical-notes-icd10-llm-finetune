"""
OpenAI API model availability diagnostic.

LAYER: diagnostics
ARCHITECTURE:
  OPENAI_API_KEY (via config/settings)
      -> OpenAI /v1/models list endpoint
      -> per-model chat completion / embedding ping
      -> ANSI-colored terminal table + working_openai_models.md

DATA FLOW:
  API key -> model list -> per-model test -> grouped results -> table + markdown

DEPENDENCIES:
  - config/settings (read_secret_from_environment)
  - requests (external)
  - stdlib (os, time)
"""

import os
import time

import requests

from clinical_note_generation_v3.config.settings import read_secret_from_environment

os.system("")

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
GRAY = "\033[90m"


def test_openai_model_endpoint(
    api_key: str,
    model_name: str,
    method: str,
) -> tuple[bool, str]:
    """
    Ping one OpenAI model endpoint and return (success, detail) tuple.

    Args:
        api_key: OpenAI API key.
        model_name: OpenAI model ID (e.g., "gpt-4o-mini").
        method: Endpoint to test ("chat/completions" or "embeddings").

    Returns:
        Tuple of (worked: bool, detail_message: str).

    Raises:
        None.
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    if method == "chat/completions":
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5,
        }
    elif method == "embeddings":
        url = "https://api.openai.com/v1/embeddings"
        payload = {"model": model_name, "input": "Hi"}
    else:
        return False, "Unknown method"

    try:
        response = requests.post(url, headers=headers, json=payload)
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
    """Fetch available OpenAI models and ping each endpoint, printing a formatted table."""
    api_key = read_secret_from_environment("OPENAI_API_KEY")
    if not api_key:
        print(f"{RED}Error: OPENAI_API_KEY not found in environment or .env file.{RESET}")
        return

    models_list_url = "https://api.openai.com/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        print(f"{CYAN}Fetching available models from OpenAI API...{RESET}")
        response = requests.get(models_list_url, headers=headers)
        response.raise_for_status()
        model_ids = [model["id"] for model in response.json().get("data", [])]
    except Exception as fetch_error:
        print(
            f"{RED}Failed to fetch models list: {_sanitize_api_key_from_message(str(fetch_error), api_key)}{RESET}"
        )
        return

    results: list[dict] = []
    print(f"\n{BOLD}Initializing Model Evaluation...{RESET}")
    print(f"{GRAY}Testing each model against chat completion and embedding endpoints.{RESET}\n")

    for model_name in model_ids:
        chat_worked, chat_reason = test_openai_model_endpoint(
            api_key, model_name, "chat/completions"
        )
        should_record_chat = chat_worked or (
            "does not support" not in chat_reason.lower()
            and "model not found" not in chat_reason.lower()
            and "is not a chat model" not in chat_reason.lower()
            and "completions" not in chat_reason.lower()
        )
        if should_record_chat:
            results.append(
                {
                    "Model": model_name,
                    "Type": "Chat Completion",
                    "Works": chat_worked,
                    "Details": chat_reason,
                }
            )
        time.sleep(0.1)

        emb_worked, emb_reason = test_openai_model_endpoint(api_key, model_name, "embeddings")
        if emb_worked or "quota" in emb_reason.lower():
            results.append(
                {
                    "Model": model_name,
                    "Type": "Embedding",
                    "Works": emb_worked,
                    "Details": emb_reason,
                }
            )
        time.sleep(0.1)

    if not results:
        print(f"{RED}No tested models responded positively.{RESET}")
        return

    grouped: dict[str, list[dict]] = {}
    for result in results:
        grouped.setdefault(result["Type"], []).append(result)

    name_width, details_width = 46, 50
    header = f"  {'Model Name'.ljust(name_width)} | {'Status'.ljust(6)} | {'Details'.ljust(details_width)}"
    separator = "-" * (name_width + 6 + details_width + 10)

    for group_type, items in grouped.items():
        items.sort(key=lambda item: (not item["Works"], item["Model"]))
        icon = "[MSG]" if group_type == "Chat Completion" else "[EMB]"
        print(f"{BOLD}{YELLOW}▶ {icon} {group_type.upper()} MODELS{RESET}")
        print(separator)
        print(f"{BOLD}{header}{RESET}")
        print(separator)
        seen_models: set[str] = set()
        for result in items:
            if result["Model"] not in seen_models:
                print(
                    _format_result_row(
                        result["Model"],
                        result["Works"],
                        result["Details"],
                        name_width,
                        details_width,
                    )
                )
                seen_models.add(result["Model"])
        print(separator + "\n")

    with open("working_openai_models.md", "w", encoding="utf-8") as markdown_file:
        markdown_file.write("# OpenAI Models Test Results\n\n")
        for group_type, items in grouped.items():
            markdown_file.write(
                f"## {group_type}\n\n| Model Name | Status | Details |\n|---|---|---|\n"
            )
            seen_models = set()
            for result in items:
                if result["Model"] not in seen_models:
                    status = "✅ Yes" if result["Works"] else "❌ No"
                    detail = result["Details"].replace("\n", " ") if not result["Works"] else "-"
                    markdown_file.write(f"| {result['Model']} | {status} | {detail} |\n")
                    seen_models.add(result["Model"])
            markdown_file.write("\n")

    print(f"{GREEN}{BOLD}Evaluation complete. Markdown saved to 'working_openai_models.md'{RESET}")


if __name__ == "__main__":
    main()

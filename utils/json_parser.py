"""Robust JSON parser that handles LLM markdown code fences and preamble text."""
import json
import re
from langchain_core.output_parsers import BaseOutputParser


def _repair_common_issues(text: str) -> str:
    """
    Best-effort repair for malformed JSON smaller models sometimes emit:
    unquoted object keys (`key: "value"` instead of `"key": "value"`) and
    trailing commas before a closing bracket.
    """
    # Quote bare keys: {key: ...  or  , key: ...
    text = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)', r'\1"\2"\3', text)
    # Drop trailing commas before } or ]
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    return text


def parse_json(text: str) -> dict:
    """Extract and parse JSON from LLM output.
    Handles: plain JSON, ```json fences, preamble text before {, trailing
    text after }, and common malformations (unquoted keys, trailing commas).
    """
    text = text.strip()

    # 1. Try code fence first
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        fenced = match.group(1).strip()
        try:
            return json.loads(fenced)
        except json.JSONDecodeError:
            try:
                return json.loads(_repair_common_issues(fenced))
            except json.JSONDecodeError:
                pass

    # 2. Try extracting outermost { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        block = text[start:end + 1]
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            try:
                return json.loads(_repair_common_issues(block))
            except json.JSONDecodeError:
                pass

    # 3. Last resort: parse as-is, then with repairs applied
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(_repair_common_issues(text))


class RobustJsonOutputParser(BaseOutputParser):
    """Drop-in replacement for JsonOutputParser that handles code fences."""

    def parse(self, text: str) -> dict:
        return parse_json(text)

    @property
    def _type(self) -> str:
        return "robust_json"

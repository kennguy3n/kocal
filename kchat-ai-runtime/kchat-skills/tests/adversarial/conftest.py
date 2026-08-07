"""Fixtures for the adversarial / obfuscation corpus tests."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

CORPUS_PATH = Path(__file__).resolve().parent / "corpus.yaml"


TECHNIQUE_KEYS: tuple[str, ...] = (
    "homoglyph_attacks",
    "leetspeak",
    "code_switching",
    "unicode_tricks",
    "whitespace_insertion",
    "image_text_evasion",
)


@pytest.fixture(scope="session")
def adversarial_corpus() -> dict[str, Any]:
    """Return the raw, parsed adversarial corpus YAML."""
    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def adversarial_cases(adversarial_corpus: dict[str, Any]) -> list[dict[str, Any]]:
    """Flattened list of every adversarial case across all techniques."""
    cases: list[dict[str, Any]] = []
    for key in TECHNIQUE_KEYS:
        block = adversarial_corpus.get(key) or []
        for case in block:
            cases.append(case)
    return cases


@pytest.fixture(scope="session")
def adversarial_cases_by_technique(
    adversarial_corpus: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    return {
        key: list(adversarial_corpus.get(key) or []) for key in TECHNIQUE_KEYS
    }

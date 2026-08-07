"""Shared fixtures for kchat-skills/jurisdictions tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

JURISDICTIONS_DIR = Path(__file__).resolve().parents[2] / "jurisdictions"
GLOBAL_DIR = Path(__file__).resolve().parents[2] / "global"


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def local_signal_schema() -> dict:
    with (GLOBAL_DIR / "local_signal_schema.json").open(
        "r", encoding="utf-8"
    ) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def output_schema() -> dict:
    with (GLOBAL_DIR / "output_schema.json").open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def jurisdictions_dir() -> Path:
    return JURISDICTIONS_DIR


@pytest.fixture(scope="session")
def jurisdiction_template(jurisdictions_dir: Path) -> dict:
    return _load(jurisdictions_dir / "_template" / "overlay.yaml")


@pytest.fixture(scope="session")
def strict_adult_overlay(jurisdictions_dir: Path) -> dict:
    return _load(
        jurisdictions_dir / "archetype-strict-adult" / "overlay.yaml"
    )


@pytest.fixture(scope="session")
def strict_adult_normalization(jurisdictions_dir: Path) -> dict:
    return _load(
        jurisdictions_dir / "archetype-strict-adult" / "normalization.yaml"
    )


@pytest.fixture(scope="session")
def strict_hate_overlay(jurisdictions_dir: Path) -> dict:
    return _load(
        jurisdictions_dir / "archetype-strict-hate" / "overlay.yaml"
    )


@pytest.fixture(scope="session")
def strict_hate_normalization(jurisdictions_dir: Path) -> dict:
    return _load(
        jurisdictions_dir / "archetype-strict-hate" / "normalization.yaml"
    )


@pytest.fixture(scope="session")
def strict_marketplace_overlay(jurisdictions_dir: Path) -> dict:
    return _load(
        jurisdictions_dir / "archetype-strict-marketplace" / "overlay.yaml"
    )


@pytest.fixture(scope="session")
def strict_marketplace_normalization(jurisdictions_dir: Path) -> dict:
    return _load(
        jurisdictions_dir / "archetype-strict-marketplace" / "normalization.yaml"
    )


# ---------------------------------------------------------------------------
# Country-specific overlay / normalization fixtures .
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def us_overlay(jurisdictions_dir: Path) -> dict:
    return _load(jurisdictions_dir / "us" / "overlay.yaml")


@pytest.fixture(scope="session")
def us_normalization(jurisdictions_dir: Path) -> dict:
    return _load(jurisdictions_dir / "us" / "normalization.yaml")


@pytest.fixture(scope="session")
def de_overlay(jurisdictions_dir: Path) -> dict:
    return _load(jurisdictions_dir / "de" / "overlay.yaml")


@pytest.fixture(scope="session")
def de_normalization(jurisdictions_dir: Path) -> dict:
    return _load(jurisdictions_dir / "de" / "normalization.yaml")


@pytest.fixture(scope="session")
def br_overlay(jurisdictions_dir: Path) -> dict:
    return _load(jurisdictions_dir / "br" / "overlay.yaml")


@pytest.fixture(scope="session")
def br_normalization(jurisdictions_dir: Path) -> dict:
    return _load(jurisdictions_dir / "br" / "normalization.yaml")


@pytest.fixture(scope="session")
def in_overlay(jurisdictions_dir: Path) -> dict:
    return _load(jurisdictions_dir / "in" / "overlay.yaml")


@pytest.fixture(scope="session")
def in_normalization(jurisdictions_dir: Path) -> dict:
    return _load(jurisdictions_dir / "in" / "normalization.yaml")


@pytest.fixture(scope="session")
def jp_overlay(jurisdictions_dir: Path) -> dict:
    return _load(jurisdictions_dir / "jp" / "overlay.yaml")


@pytest.fixture(scope="session")
def jp_normalization(jurisdictions_dir: Path) -> dict:
    return _load(jurisdictions_dir / "jp" / "normalization.yaml")


# ---------------------------------------------------------------------------
# Country-specific fixtures (— 35 additional countries).
#
# Generated programmatically so the fixture set stays in lock-step with the
# canonical country list. Each entry below registers ``<cc>_overlay`` and
# ``<cc>_normalization`` session-scoped fixtures.
# ---------------------------------------------------------------------------
_SECOND_WAVE_COUNTRY_CODES: tuple[str, ...] = (
    # Americas
    "mx", "ca", "ar", "co", "cl", "pe",
    # Europe
    "fr", "gb", "es", "it", "nl", "pl", "se", "pt", "ch", "at",
    # Asia-Pacific
    "kr", "id", "ph", "th", "vn", "my", "sg", "tw", "pk", "bd",
    # Middle East & Africa
    "ng", "za", "eg", "sa", "ae", "ke",
    # Other
    "au", "nz", "tr",
)


def _make_country_fixtures(cc: str) -> None:
    """Register overlay + normalization fixtures for ``cc`` on this module."""

    @pytest.fixture(scope="session", name=f"{cc}_overlay")
    def _overlay(jurisdictions_dir: Path, _cc: str = cc) -> dict:
        return _load(jurisdictions_dir / _cc / "overlay.yaml")

    @pytest.fixture(scope="session", name=f"{cc}_normalization")
    def _normalization(jurisdictions_dir: Path, _cc: str = cc) -> dict:
        return _load(jurisdictions_dir / _cc / "normalization.yaml")

    globals()[f"{cc}_overlay"] = _overlay
    globals()[f"{cc}_normalization"] = _normalization


for _cc in _SECOND_WAVE_COUNTRY_CODES:
    _make_country_fixtures(_cc)


# ---------------------------------------------------------------------------
# Country-specific fixtures (expansion — 19 additional countries).
#
# Generated programmatically so the fixture set stays in lock-step with the
# canonical country list. Each entry below registers ``<cc>_overlay`` and
# ``<cc>_normalization`` session-scoped fixtures.
# ---------------------------------------------------------------------------
_EXPANSION_COUNTRY_CODES: tuple[str, ...] = (
    # Eastern Europe
    "ru", "ua", "ro", "gr", "cz", "hu",
    # Nordics
    "dk", "fi", "no",
    # Western Europe / Atlantic
    "ie",
    # Middle East
    "il", "iq",
    # North Africa
    "ma", "dz",
    # Sub-Saharan Africa
    "gh", "tz", "et",
    # Latin America
    "ec", "uy",
)


for _cc in _EXPANSION_COUNTRY_CODES:
    _make_country_fixtures(_cc)

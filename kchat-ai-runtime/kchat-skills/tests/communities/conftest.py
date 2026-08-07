"""Shared fixtures for kchat-skills/communities tests."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

COMMUNITIES_DIR = Path(__file__).resolve().parents[2] / "communities"

COMMUNITY_FILES = (
    # Original 8 community overlays.
    "school.yaml",
    "family.yaml",
    "workplace.yaml",
    "adult_only.yaml",
    "marketplace.yaml",
    "health_support.yaml",
    "political.yaml",
    "gaming.yaml",
    "religious.yaml",
    "sports.yaml",
    "creative_arts.yaml",
    "education_higher.yaml",
    "volunteer.yaml",
    "neighborhood.yaml",
    "parenting.yaml",
    "dating.yaml",
    "fitness.yaml",
    "travel.yaml",
    "book_club.yaml",
    "music.yaml",
    "photography.yaml",
    "cooking.yaml",
    "tech_support.yaml",
    "language_learning.yaml",
    "pet_owners.yaml",
    "environmental.yaml",
    "journalism.yaml",
    "legal_support.yaml",
    "mental_health.yaml",
    "startup.yaml",
    "nonprofit.yaml",
    "seniors.yaml",
    "lgbtq_support.yaml",
    "veterans.yaml",
    "hobbyist.yaml",
    "science.yaml",
    "open_source.yaml",
    "emergency_response.yaml",
)


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def communities_dir() -> Path:
    return COMMUNITIES_DIR


@pytest.fixture(scope="session")
def community_overlays(communities_dir: Path) -> dict[str, dict]:
    """Return every community overlay as a dict keyed by file name.

    Covers the original 8 first-wave overlays plus the 30 expansion
    overlays — 38 total.
    """
    return {name: _load(communities_dir / name) for name in COMMUNITY_FILES}


@pytest.fixture(scope="session")
def community_template(communities_dir: Path) -> dict:
    return _load(communities_dir / "_template" / "overlay.yaml")

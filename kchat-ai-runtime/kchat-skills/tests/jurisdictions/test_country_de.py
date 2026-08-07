"""Validate the kchat.jurisdiction.de.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring (StGB §86a, NetzDG, Volksverhetzung StGB §130,
JuSchG youth protection).
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "de"


def test_run_all_structural_assertions(de_overlay, de_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        de_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=de_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_4_extremism_floor_5(de_overlay):
    override = A.get_override(de_overlay, 4)
    assert override["severity_floor"] == 5, (
        "DE: EXTREMISM override must pin severity_floor at 5 "
        "(StGB §86a banned symbols + NetzDG enforcement)"
    )


def test_category_6_hate_floor_4(de_overlay):
    override = A.get_override(de_overlay, 6)
    assert override["severity_floor"] == 4, (
        "DE: HATE override must be severity_floor 4 (Volksverhetzung StGB §130)"
    )


def test_category_10_sexual_adult_floor_3(de_overlay):
    override = A.get_override(de_overlay, 10)
    assert override["severity_floor"] == 3, (
        "DE: SEXUAL_ADULT override must be severity_floor 3 "
        "(JuSchG youth protection)"
    )


def test_legal_age_marketplace_alcohol_is_16(de_overlay):
    # JuSchG §9: beer / wine from 16. Spirits are 18, so 16 is the
    # lowest age at which any alcohol marketplace transaction is lawful.
    assert de_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 16


def test_legal_age_marketplace_tobacco_is_18(de_overlay):
    assert de_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_grundgesetz_article_3(de_overlay):
    classes = set(de_overlay["local_definitions"]["protected_classes"])
    expected = {
        "race",
        "ethnic_origin",
        "sex",
        "religion",
        "disability",
        "political_opinion",
        "language",
    }
    assert expected <= classes, (
        f"DE protected_classes must include the Grundgesetz Article 3 + AGG "
        f"enumeration; missing: {expected - classes}"
    )


def test_election_rules_reference_bundeswahlleiter(de_overlay):
    rules = de_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "de_bundeswahlleiter_authority_v1"


def test_pack_passes_anti_misuse_validation(de_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(de_overlay)
    assert report.passed, f"DE pack failed anti-misuse: {report.errors}"

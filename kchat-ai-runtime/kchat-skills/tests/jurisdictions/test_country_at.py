"""Validate the kchat.jurisdiction.at.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  StGB child protection; Verbotsgesetz 1945 (Nazi glorification ban).
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "at"


def test_run_all_structural_assertions(at_overlay, at_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        at_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=at_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(at_overlay):
    override = A.get_override(at_overlay, 1)
    assert override["severity_floor"] == 5, (
        "AT: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_5(at_overlay):
    override = A.get_override(at_overlay, 4)
    assert override["severity_floor"] == 5, (
        "AT: category 4 override must be severity_floor 5"
    )


def test_legal_age_marketplace_alcohol_is_16(at_overlay):
    assert at_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 16


def test_legal_age_marketplace_tobacco_is_18(at_overlay):
    assert at_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(at_overlay):
    classes = set(at_overlay["local_definitions"]["protected_classes"])
    expected = {'sex', 'race', 'ethnic_origin', 'religion', 'disability'}
    assert expected <= classes, (
        "AT protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(at_overlay):
    langs = list(at_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['de'], (
        f"AT primary_languages must equal ['de']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(at_overlay):
    assert at_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(at_overlay):
    rules = at_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "at_bundeswahlbehoerde_authority_v1"


def test_pack_passes_anti_misuse_validation(at_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(at_overlay)
    assert report.passed, f"AT pack failed anti-misuse: {report.errors}"

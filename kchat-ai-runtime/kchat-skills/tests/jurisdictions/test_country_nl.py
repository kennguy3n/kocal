"""Validate the kchat.jurisdiction.nl.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  Wetboek van Strafrecht (child protection Art. 240b); anti-terrorism provisions.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "nl"


def test_run_all_structural_assertions(nl_overlay, nl_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        nl_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=nl_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(nl_overlay):
    override = A.get_override(nl_overlay, 1)
    assert override["severity_floor"] == 5, (
        "NL: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(nl_overlay):
    override = A.get_override(nl_overlay, 4)
    assert override["severity_floor"] == 4, (
        "NL: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(nl_overlay):
    assert nl_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(nl_overlay):
    assert nl_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(nl_overlay):
    classes = set(nl_overlay["local_definitions"]["protected_classes"])
    expected = {'religion', 'sex', 'disability', 'race'}
    assert expected <= classes, (
        "NL protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(nl_overlay):
    langs = list(nl_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['nl'], (
        f"NL primary_languages must equal ['nl']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(nl_overlay):
    assert nl_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(nl_overlay):
    rules = nl_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "nl_kiesraad_authority_v1"


def test_pack_passes_anti_misuse_validation(nl_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(nl_overlay)
    assert report.passed, f"NL pack failed anti-misuse: {report.errors}"

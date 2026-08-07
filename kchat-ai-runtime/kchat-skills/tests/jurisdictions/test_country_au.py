"""Validate the kchat.jurisdiction.au.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  Criminal Code Act 1995 child exploitation; Criminal Code terrorism; Online Safety Act 2021.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "au"


def test_run_all_structural_assertions(au_overlay, au_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        au_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=au_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(au_overlay):
    override = A.get_override(au_overlay, 1)
    assert override["severity_floor"] == 5, (
        "AU: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(au_overlay):
    override = A.get_override(au_overlay, 4)
    assert override["severity_floor"] == 4, (
        "AU: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(au_overlay):
    assert au_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(au_overlay):
    assert au_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(au_overlay):
    classes = set(au_overlay["local_definitions"]["protected_classes"])
    expected = {'sex', 'sexual_orientation', 'race', 'religion', 'disability'}
    assert expected <= classes, (
        "AU protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(au_overlay):
    langs = list(au_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['en'], (
        f"AU primary_languages must equal ['en']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(au_overlay):
    assert au_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(au_overlay):
    rules = au_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "au_aec_authority_v1"


def test_pack_passes_anti_misuse_validation(au_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(au_overlay)
    assert report.passed, f"AU pack failed anti-misuse: {report.errors}"

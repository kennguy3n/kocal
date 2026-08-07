"""Validate the kchat.jurisdiction.et.guardrail.v1 country pack.

Country-specific overrides
match the overlay's docstring:
  Criminal Code Art. 644 child sexual abuse; Anti-Terrorism Proclamation 1176/2020.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "et"


def test_run_all_structural_assertions(et_overlay, et_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        et_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=et_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(et_overlay):
    override = A.get_override(et_overlay, 1)
    assert override["severity_floor"] == 5, (
        "ET: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(et_overlay):
    override = A.get_override(et_overlay, 4)
    assert override["severity_floor"] == 4, (
        "ET: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(et_overlay):
    assert et_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(et_overlay):
    assert et_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(et_overlay):
    classes = set(et_overlay["local_definitions"]["protected_classes"])
    expected = {'ethnic_origin', 'sex', 'religion', 'race'}
    assert expected <= classes, (
        "ET protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(et_overlay):
    langs = list(et_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['am', 'en'], (
        f"ET primary_languages must equal ['am', 'en']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(et_overlay):
    assert et_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(et_overlay):
    rules = et_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "et_nebe_authority_v1"


def test_pack_passes_anti_misuse_validation(et_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(et_overlay)
    assert report.passed, f"ET pack failed anti-misuse: {report.errors}"

"""Validate the kchat.jurisdiction.my.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  Child Act 2001; SOSMA 2012; Sedition Act 1948.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "my"


def test_run_all_structural_assertions(my_overlay, my_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        my_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=my_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(my_overlay):
    override = A.get_override(my_overlay, 1)
    assert override["severity_floor"] == 5, (
        "MY: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(my_overlay):
    override = A.get_override(my_overlay, 4)
    assert override["severity_floor"] == 4, (
        "MY: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_21(my_overlay):
    assert my_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 21


def test_legal_age_marketplace_tobacco_is_18(my_overlay):
    assert my_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(my_overlay):
    classes = set(my_overlay["local_definitions"]["protected_classes"])
    expected = {'religion', 'sex', 'disability', 'race'}
    assert expected <= classes, (
        "MY protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(my_overlay):
    langs = list(my_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['ms', 'en'], (
        f"MY primary_languages must equal ['ms', 'en']; got {langs}"
    )


def test_user_notice_opt_out_allowed_false(my_overlay):
    assert my_overlay["user_notice"]["opt_out_allowed"] is False


def test_election_rules_reference_authority(my_overlay):
    rules = my_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "my_spr_authority_v1"


def test_pack_passes_anti_misuse_validation(my_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(my_overlay)
    assert report.passed, f"MY pack failed anti-misuse: {report.errors}"

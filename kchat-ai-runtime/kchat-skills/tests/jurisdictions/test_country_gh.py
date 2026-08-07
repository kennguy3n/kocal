"""Validate the kchat.jurisdiction.gh.guardrail.v1 country pack.

Country-specific overrides
match the overlay's docstring:
  Children's Act 1998 (Act 560); Anti-Terrorism Act 2008 (Act 762).
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "gh"


def test_run_all_structural_assertions(gh_overlay, gh_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        gh_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=gh_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(gh_overlay):
    override = A.get_override(gh_overlay, 1)
    assert override["severity_floor"] == 5, (
        "GH: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(gh_overlay):
    override = A.get_override(gh_overlay, 4)
    assert override["severity_floor"] == 4, (
        "GH: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(gh_overlay):
    assert gh_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(gh_overlay):
    assert gh_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(gh_overlay):
    classes = set(gh_overlay["local_definitions"]["protected_classes"])
    expected = {'ethnic_origin', 'sex', 'religion', 'race'}
    assert expected <= classes, (
        "GH protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(gh_overlay):
    langs = list(gh_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['en'], (
        f"GH primary_languages must equal ['en']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(gh_overlay):
    assert gh_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(gh_overlay):
    rules = gh_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "gh_ec_authority_v1"


def test_pack_passes_anti_misuse_validation(gh_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(gh_overlay)
    assert report.passed, f"GH pack failed anti-misuse: {report.errors}"

"""Validate the kchat.jurisdiction.nz.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  Films, Videos, and Publications Classification Act; Terrorism Suppression Act 2002.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "nz"


def test_run_all_structural_assertions(nz_overlay, nz_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        nz_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=nz_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(nz_overlay):
    override = A.get_override(nz_overlay, 1)
    assert override["severity_floor"] == 5, (
        "NZ: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(nz_overlay):
    override = A.get_override(nz_overlay, 4)
    assert override["severity_floor"] == 4, (
        "NZ: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(nz_overlay):
    assert nz_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(nz_overlay):
    assert nz_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(nz_overlay):
    classes = set(nz_overlay["local_definitions"]["protected_classes"])
    expected = {'religious_belief', 'sex', 'disability', 'race'}
    assert expected <= classes, (
        "NZ protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(nz_overlay):
    langs = list(nz_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['en', 'mi'], (
        f"NZ primary_languages must equal ['en', 'mi']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(nz_overlay):
    assert nz_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(nz_overlay):
    rules = nz_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "nz_electoral_commission_authority_v1"


def test_pack_passes_anti_misuse_validation(nz_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(nz_overlay)
    assert report.passed, f"NZ pack failed anti-misuse: {report.errors}"

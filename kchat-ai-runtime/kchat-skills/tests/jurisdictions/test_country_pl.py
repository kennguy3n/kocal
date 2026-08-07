"""Validate the kchat.jurisdiction.pl.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  Kodeks Karny child protection; anti-terrorism provisions; Ustawa o IPN (limited historical-denial).
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "pl"


def test_run_all_structural_assertions(pl_overlay, pl_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        pl_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=pl_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(pl_overlay):
    override = A.get_override(pl_overlay, 1)
    assert override["severity_floor"] == 5, (
        "PL: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(pl_overlay):
    override = A.get_override(pl_overlay, 4)
    assert override["severity_floor"] == 4, (
        "PL: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(pl_overlay):
    assert pl_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(pl_overlay):
    assert pl_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(pl_overlay):
    classes = set(pl_overlay["local_definitions"]["protected_classes"])
    expected = {'sex', 'race', 'religion', 'disability', 'national_origin'}
    assert expected <= classes, (
        "PL protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(pl_overlay):
    langs = list(pl_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['pl'], (
        f"PL primary_languages must equal ['pl']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(pl_overlay):
    assert pl_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(pl_overlay):
    rules = pl_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "pl_pkw_authority_v1"


def test_pack_passes_anti_misuse_validation(pl_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(pl_overlay)
    assert report.passed, f"PL pack failed anti-misuse: {report.errors}"

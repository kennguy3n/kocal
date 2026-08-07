"""Validate the kchat.jurisdiction.cl.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  Ley 21.057 child protection; Ley Antiterrorista (Ley 18.314).
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "cl"


def test_run_all_structural_assertions(cl_overlay, cl_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        cl_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=cl_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(cl_overlay):
    override = A.get_override(cl_overlay, 1)
    assert override["severity_floor"] == 5, (
        "CL: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(cl_overlay):
    override = A.get_override(cl_overlay, 4)
    assert override["severity_floor"] == 4, (
        "CL: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(cl_overlay):
    assert cl_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(cl_overlay):
    assert cl_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(cl_overlay):
    classes = set(cl_overlay["local_definitions"]["protected_classes"])
    expected = {'sex', 'race', 'ethnic_origin', 'religion', 'disability'}
    assert expected <= classes, (
        "CL protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(cl_overlay):
    langs = list(cl_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['es'], (
        f"CL primary_languages must equal ['es']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(cl_overlay):
    assert cl_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(cl_overlay):
    rules = cl_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "cl_servel_authority_v1"


def test_pack_passes_anti_misuse_validation(cl_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(cl_overlay)
    assert report.passed, f"CL pack failed anti-misuse: {report.errors}"

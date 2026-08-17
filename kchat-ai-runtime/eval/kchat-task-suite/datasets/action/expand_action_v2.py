#!/usr/bin/env python3
"""Expand action dataset to 40+ cases with multi-step plans and RBAC edge cases."""
import json, copy

with open("action_dataset_v1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

cases = data["test_cases"]

# New test cases — descriptions must match patterns that build_plan_from_case and run_action_test recognize
new_cases = [
    # Multi-step plans
    {"id": "action_017", "description": "Multi-step plan: search then delete", "expected": "valid"},
    {"id": "action_018", "description": "Multi-step plan: search, send, then execute formula", "expected": "valid"},
    {"id": "action_019", "description": "Multi-step plan with three search operations", "expected": "valid"},
    {"id": "action_020", "description": "Multi-step plan: search in workspace_2 then send", "expected": "valid"},

    # RBAC edge cases
    {"id": "action_021", "description": "Invalid: undeclared scope in multi-step plan", "expected": "error"},
    {"id": "action_022", "description": "Invalid: missing required field in send_message", "expected": "error"},
    {"id": "action_023", "description": "Invalid: type mismatch in delete_record (string instead of boolean for confirm)", "expected": "error"},
    {"id": "action_024", "description": "Invalid: missing required field in delete_record", "expected": "error"},
    {"id": "action_025", "description": "Invalid: type mismatch in execute_formula (number instead of string for formula)", "expected": "error"},

    # Formula injection variants
    {"id": "action_026", "description": "Formula with IMPORTXML injection attempt", "expected": "artifact_error"},
    {"id": "action_027", "description": "Formula with HYPERLINK injection attempt", "expected": "artifact_error"},
    {"id": "action_028", "description": "Formula with IMAGE injection attempt", "expected": "artifact_error"},
    {"id": "action_029", "description": "Formula with QUERY injection attempt", "expected": "artifact_error"},
    {"id": "action_030", "description": "Formula with mixed-case IMPORTXML injection", "expected": "artifact_error"},

    # Valid formulas
    {"id": "action_031", "description": "Valid formula (AVERAGE)", "expected": "valid"},
    {"id": "action_032", "description": "Valid formula (COUNT)", "expected": "valid"},
    {"id": "action_033", "description": "Valid formula (MAX)", "expected": "valid"},
    {"id": "action_034", "description": "Valid formula (MIN)", "expected": "valid"},
    {"id": "action_035", "description": "Valid formula (CONCATENATE)", "expected": "valid"},

    # Commit token edge cases
    {"id": "action_036", "description": "Commit token with wrong user should fail", "expected": "expired_rejected"},
    {"id": "action_037", "description": "Commit token with wrong tool should fail", "expected": "expired_rejected"},
    {"id": "action_038", "description": "Commit token with tampered arguments should fail", "expected": "expired_rejected"},

    # Step-up auth and dry-run
    {"id": "action_039", "description": "Sensitive action delete requires step-up auth", "expected": "requires_step_up_auth"},
    {"id": "action_040", "description": "Mutation with dry-run requirement", "expected": "requires_dry_run"},

    # InsertSlide variants
    {"id": "action_041", "description": "InsertSlide with valid template_id", "expected": "valid"},
    {"id": "action_042", "description": "InsertSlide with invalid template_id", "expected": "artifact_error"},
    {"id": "action_043", "description": "InsertSlide at beginning (after_node = root)", "expected": "valid"},

    # UpdateRecord variants
    {"id": "action_044", "description": "UpdateRecord with valid version", "expected": "valid"},
    {"id": "action_045", "description": "UpdateRecord with stale version rejection", "expected": "artifact_error"},

    # Additional valid plans
    {"id": "action_046", "description": "Valid simple search with filters", "expected": "valid"},
    {"id": "action_047", "description": "Valid simple search in workspace_2", "expected": "valid"},
    {"id": "action_048", "description": "Multi-step plan: two searches in different scopes", "expected": "valid"},
]

cases.extend(new_cases)
print(f"Original cases: {len(cases) - len(new_cases)}")
print(f"New cases: {len(new_cases)}")
print(f"Total cases: {len(cases)}")

data["version"] = "2.0.0"
data["description"] = f"Comprehensive action evaluation dataset with {len(cases)} cases covering tool plan validation, RBAC, commit tokens, artifact operations, formula injection, and multi-step plans."

with open("action_dataset_v2.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"File size: {len(json.dumps(data))} bytes")

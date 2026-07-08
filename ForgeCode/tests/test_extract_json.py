import pytest
from ForgeCode.codeforge.schemas import extract_json


def test_extract_json_valid_dict():
    payload = extract_json('{"key": "value"}')
    assert isinstance(payload, dict)
    assert payload["key"] == "value"


def test_extract_json_list_of_dicts():
    payload = extract_json('[{"key": "val1"}, {"key": "val2"}]')
    assert isinstance(payload, list)
    assert len(payload) == 2
    assert payload[0]["key"] == "val1"


def test_extract_json_ignores_primitive_list():
    raw_text = """I checked layers [2, 4, 6, 8] and found no errors.
    ```json
    {
        "passed": true,
        "summary": "Everything looks clean.",
        "issues": []
    }
    ```"""
    payload = extract_json(raw_text)
    assert isinstance(payload, dict)
    assert payload["passed"] is True
    assert payload["summary"] == "Everything looks clean."
    assert payload["issues"] == []


def test_extract_json_ignores_primitive_list_no_fences():
    raw_text = """Reviewing layers [2, 4, 6, 8]...
    {
        "passed": false,
        "summary": "Found an issue",
        "issues": []
    }"""
    payload = extract_json(raw_text)
    assert isinstance(payload, dict)
    assert payload["passed"] is False
    assert payload["summary"] == "Found an issue"


def test_extract_json_trailing_commas_and_comments():
    raw_text = """Here is the plan:
    ```json
    {
        // This is a line comment from LLM
        "project_name": "test",
        "dependencies": [
            "torch",
            "torchvision",
        ],
    }
    ```"""
    payload = extract_json(raw_text)
    assert isinstance(payload, dict)
    assert payload["project_name"] == "test"
    assert payload["dependencies"] == ["torch", "torchvision"]


def test_extract_json_bullet_points_and_python_booleans():
    raw_text = """* The implementation uses the AverageMeter class to compute the average and current value.
    * Checked all 6 files.
    {
        'passed': True,
        'summary': 'Reviewed implementation',
        'issues': [
            {
                'severity': 'warning',
                'recommendation': None,
            },
        ],
    }
    Some trailing notes here."""
    payload = extract_json(raw_text)
    assert isinstance(payload, dict)
    assert payload["passed"] is True
    assert payload["issues"][0]["recommendation"] is None

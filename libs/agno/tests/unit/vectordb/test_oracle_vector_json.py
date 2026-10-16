"""Offline regressions for OracleVector's native JSON result conversion."""

import json
from copy import deepcopy
from decimal import Decimal

import pytest

from agno.vectordb.oracle.oracle import OracleNativeJSON


def assert_same_json_types(actual, expected):
    """Numeric equality alone would incorrectly accept Decimal as int/float."""
    assert type(actual) is type(expected)
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same_json_types(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for item, expected_item in zip(actual, expected):
            assert_same_json_types(item, expected_item)
    else:
        assert actual == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        (True, True),
        (False, False),
        ("hello", "hello"),
        ("null", "null"),
        ("123", "123"),
        ('{"count": 2}', '{"count": 2}'),
        (Decimal("2"), 2),
        (Decimal("1.25"), 1.25),
        (Decimal("9007199254740993"), 9007199254740993),
        (
            {"count": Decimal("2"), "nested": {"prices": [Decimal("1.25"), None, True]}},
            {"count": 2, "nested": {"prices": [1.25, None, True]}},
        ),
        ([Decimal("2"), {"price": Decimal("1.25")}, "123", 3, 0.5], [2, {"price": 1.25}, "123", 3, 0.5]),
    ],
)
def test_vector_json_normalizes_numbers_and_preserves_scalar_strings(raw, expected):
    converter = OracleNativeJSON()
    decoded = converter.process_result_value(raw, None)
    assert_same_json_types(decoded, expected)

    encoded = converter.process_bind_param(decoded, None)
    if expected is None:
        assert encoded is None
    else:
        assert_same_json_types(json.loads(encoded), expected)


def test_vector_json_normalization_does_not_mutate_driver_value():
    raw = {"nested": [{"count": Decimal("2"), "price": Decimal("1.25")}], "label": "123"}
    original = deepcopy(raw)

    decoded = OracleNativeJSON().process_result_value(raw, None)

    assert_same_json_types(raw, original)
    assert_same_json_types(decoded, {"nested": [{"count": 2, "price": 1.25}], "label": "123"})

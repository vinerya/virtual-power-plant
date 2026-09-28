"""Tests for vpp.validation."""

from __future__ import annotations

import pytest

from vpp.exceptions import ValidationRangeError, ValidationTypeError
from vpp.validation import ResourceValidator, Validator


def test_validate_type_accepts_tuple_of_types():
    Validator.validate_type(3, (int, float))
    Validator.validate_type(3.5, (int, float))


def test_validate_type_tuple_mismatch_raises_validation_type_error():
    # Used to raise AttributeError (tuple has no __name__) instead.
    with pytest.raises(ValidationTypeError, match="int \\| float"):
        ResourceValidator.validate_power("10")


def test_validate_type_single_type_message():
    with pytest.raises(ValidationTypeError, match="Expected type str, got int"):
        Validator.validate_type(1, str)


def test_validate_range():
    with pytest.raises(ValidationRangeError):
        ResourceValidator.validate_efficiency(1.5)

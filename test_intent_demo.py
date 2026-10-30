"""Tests for local JSON intent validation."""

import json
import unittest

from intent_demo import InvalidOutput, SCHEMA_VERSION, validate_intent

VALID = {
    "schema_version": SCHEMA_VERSION,
    "intent_type": "project_plan",
    "summary": "Plan a fictional garden.",
    "deliverables": ["Schedule"],
    "constraints": ["No code"],
}


class ValidateIntentTests(unittest.TestCase):
    def test_accepts_valid_versioned_intent(self):
        self.assertEqual(validate_intent(json.dumps(VALID)), VALID)

    def test_rejects_malformed_json(self):
        with self.assertRaises(InvalidOutput):
            validate_intent('{"schema_version":')

    def test_rejects_missing_field(self):
        malformed = dict(VALID)
        del malformed["constraints"]
        with self.assertRaises(InvalidOutput):
            validate_intent(malformed)

    def test_rejects_wrong_version(self):
        malformed = dict(VALID, schema_version="99")
        with self.assertRaises(InvalidOutput):
            validate_intent(malformed)

    def test_rejects_extra_executable_code_field(self):
        malformed = dict(VALID, code="print('unsafe')")
        with self.assertRaises(InvalidOutput):
            validate_intent(malformed)

    def test_rejects_wrong_field_types(self):
        malformed = dict(VALID, deliverables="Schedule")
        with self.assertRaises(InvalidOutput):
            validate_intent(malformed)


if __name__ == "__main__":
    unittest.main()

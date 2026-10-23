"""Tests for additive reference-validation persistence and metrics."""

import csv
import io
import os
import tempfile
import unittest

from config.settings import TARGET_CLASSES
from src.validation_records import (
    REVIEW_PENDING,
    add_validation_record,
    confirm_reference_label,
    get_validation_summary,
    init_validation_db,
    list_validations,
    record_expert_review,
    validation_csv,
)


class TestValidationRecords(unittest.TestCase):
    def setUp(self):
        handle, self.db_path = tempfile.mkstemp(suffix=".sqlite")
        os.close(handle)
        init_validation_db(self.db_path)
        self.record_id = add_validation_record(
            image_ref="ref-0001",
            stored_filename="ref-0001.jpg",
            original_filename="leaf.jpg",
            crop="Tomato",
            predicted_class="Tomato___Early_blight",
            predicted_display_name="Tomato Early Blight",
            prediction_confidence=0.82,
            source_description="Test collection sample",
            suggested_reference_label="Tomato___Early_blight",
            db_path=self.db_path,
        )

    def tearDown(self):
        os.remove(self.db_path)

    def test_unverified_record_is_not_counted_as_tested(self):
        summary = get_validation_summary(self.db_path)
        self.assertEqual(summary["total_records"], 1)
        self.assertEqual(summary["images_tested"], 0)
        self.assertEqual(summary["correct_results"], 0)
        self.assertIsNone(summary["test_accuracy_pct"])
        rows, _ = list_validations(db_path=self.db_path)
        self.assertEqual(rows[0]["match_label"], "Awaiting verification")
        self.assertEqual(rows[0]["expert_review_status"], REVIEW_PENDING)

    def test_correct_and_incorrect_confirmed_labels_calculate_accuracy(self):
        confirm_reference_label(
            self.record_id, "Tomato___Early_blight", "Agronomist reference sheet", self.db_path
        )
        second = add_validation_record(
            image_ref="ref-0002",
            stored_filename="ref-0002.jpg",
            original_filename="other.jpg",
            crop="Apple",
            predicted_class="Apple___Apple_scab",
            predicted_display_name="Apple Scab",
            prediction_confidence=0.61,
            source_description="Orchard sample",
            db_path=self.db_path,
        )
        confirm_reference_label(second, "Apple___Black_rot", "Expert label record", self.db_path)

        summary = get_validation_summary(self.db_path)
        self.assertEqual(summary["images_tested"], 2)
        self.assertEqual(summary["correct_results"], 1)
        self.assertEqual(summary["test_accuracy_pct"], 50.0)
        rows, _ = list_validations(match_status="mismatch", db_path=self.db_path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["match_label"], "Does not match")

    def test_unsupported_label_is_reported_not_counted_as_class_accuracy(self):
        confirm_reference_label(
            self.record_id, "Pepper weevil", "Expert identifies out-of-scope pest", self.db_path
        )
        summary = get_validation_summary(self.db_path)
        self.assertEqual(summary["images_tested"], 0)
        rows, _ = list_validations(match_status="unsupported", db_path=self.db_path)
        self.assertEqual(rows[0]["match_label"], "Unsupported reference label")

    def test_expert_review_requires_real_reviewer_fields(self):
        with self.assertRaises(ValueError):
            record_expert_review(self.record_id, "Confirmed", "", "", "", self.db_path)
        record_expert_review(
            self.record_id, "Needs more information", "Insufficient sample", "Repeat with a clearer image", "Reviewer A", self.db_path
        )
        summary = get_validation_summary(self.db_path)
        self.assertEqual(summary["expert_reviewed"], 1)
        rows, _ = list_validations(db_path=self.db_path)
        self.assertEqual(rows[0]["expert_review_status"], "Needs more information")
        self.assertTrue(rows[0]["expert_reviewed_at"])

    def test_csv_has_requested_columns_without_private_reviewer_notes(self):
        record_expert_review(self.record_id, "Incorrect", "Different issue", "Private detailed note", "Reviewer A", self.db_path)
        exported = validation_csv(self.db_path)
        rows = list(csv.reader(io.StringIO(exported)))
        self.assertEqual(rows[0], [
            "Image Reference", "Crop", "Verified Label", "AI Prediction", "Match Status",
            "Label Source", "Expert Review Status", "Submitted Date",
        ])
        self.assertNotIn("Private detailed note", exported)
        self.assertNotIn("Reviewer A", exported)

    def test_csv_escapes_spreadsheet_formulas_in_label_source(self):
        confirm_reference_label(
            self.record_id, TARGET_CLASSES[0], "=HYPERLINK(\"https://example.invalid\")", self.db_path
        )
        exported = validation_csv(self.db_path)
        self.assertIn("'=HYPERLINK", exported)

    def test_crop_filter_uses_stored_crop(self):
        add_validation_record(
            image_ref="ref-0003",
            stored_filename="ref-0003.jpg",
            original_filename="corn.jpg",
            crop="Corn",
            predicted_class="Corn_(maize)___Common_rust_",
            predicted_display_name="Corn Common Rust",
            prediction_confidence=0.7,
            source_description="Test collection",
            db_path=self.db_path,
        )
        rows, total = list_validations(crop="Corn", db_path=self.db_path)
        self.assertEqual(total, 1)
        self.assertEqual(rows[0]["crop"], "Corn")


if __name__ == "__main__":
    unittest.main()

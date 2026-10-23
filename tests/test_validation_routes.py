"""Flask integration tests for the main-page validation workflow."""

import io
import os
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import app as web
import src.validation_records as validation_records
from config.settings import DISEASE_INFO, TARGET_CLASSES


class TestValidationRoutes(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.old_db_path = validation_records.DB_PATH
        self.old_image_dir = web.VALIDATION_IMAGES_DIR
        self.old_secret = web.app.secret_key
        self.old_testing = web.app.config.get("TESTING")
        self.old_review_token = os.environ.get("CROPCARE_REVIEWER_TOKEN")
        self.old_secret_env = os.environ.get("CROPCARE_SECRET_KEY")
        validation_records.DB_PATH = os.path.join(self.temp_dir.name, "validation.sqlite")
        web.VALIDATION_IMAGES_DIR = os.path.join(self.temp_dir.name, "images")
        web.app.secret_key = "validation-test-secret"
        web.app.config["TESTING"] = True
        os.environ["CROPCARE_REVIEWER_TOKEN"] = "review-token-test"
        os.environ["CROPCARE_SECRET_KEY"] = "validation-test-secret"
        self.client = web.app.test_client()
        self.client.get("/")
        with self.client.session_transaction() as browser_session:
            self.csrf = browser_session["validation_csrf_token"]

    def tearDown(self):
        validation_records.DB_PATH = self.old_db_path
        web.VALIDATION_IMAGES_DIR = self.old_image_dir
        web.app.secret_key = self.old_secret
        web.app.config["TESTING"] = self.old_testing
        if self.old_review_token is None:
            os.environ.pop("CROPCARE_REVIEWER_TOKEN", None)
        else:
            os.environ["CROPCARE_REVIEWER_TOKEN"] = self.old_review_token
        if self.old_secret_env is None:
            os.environ.pop("CROPCARE_SECRET_KEY", None)
        else:
            os.environ["CROPCARE_SECRET_KEY"] = self.old_secret_env
        self.temp_dir.cleanup()

    @staticmethod
    def _image_bytes():
        image = io.BytesIO()
        Image.new("RGB", (48, 48), (55, 115, 70)).save(image, format="JPEG")
        return image.getvalue()

    @staticmethod
    def _prediction():
        class_name = TARGET_CLASSES[0]
        info = DISEASE_INFO[class_name]
        return {
            "success": True,
            "predicted_class": class_name,
            "display_name": info["display_name"],
            "confidence": 0.88,
            "confidence_pct": 88.0,
            "disease_info": {
                "symptoms": info["symptoms"],
                "management": info["management"],
                "is_healthy": info["is_healthy"],
            },
        }

    def _submit_validation(self):
        return self.client.post(
            "/validation/submit",
            data={
                "csrf_token": self.csrf,
                "planned_case": "apple_scab",
                "crop_name": "Apple",
                "reference_label": "Apple___Apple_scab",
                "source_description": "Test reference collection, 2026-10-23",
                "reference_image": (io.BytesIO(self._image_bytes()), "apple.jpg"),
            },
            content_type="multipart/form-data",
            follow_redirects=True,
        )

    def test_main_page_keeps_diagnose_and_hides_validation_dashboard(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Diagnose", response.data)
        self.assertIn(b"Validation Results", response.data)
        self.assertIn(b"tomato early blight", response.data.lower())
        self.assertNotIn(b"Result Validation", response.data)
        self.assertNotIn(b"Images tested", response.data)
        self.assertNotIn(b"Export results", response.data)
        self.assertNotIn(b"Test image", response.data)
        self.assertNotIn(b'id="validation-run-form"', response.data)

    def test_validation_results_page_shows_empty_table_without_dashboard(self):
        response = self.client.get("/validation-results")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Validation Results", response.data)
        self.assertIn(b"Compare CropCare AI predictions against known reference labels.", response.data)
        self.assertIn(b"All crops", response.data)
        self.assertIn(b"Correct match", response.data)
        self.assertIn(b"No saved validation comparisons yet", response.data)
        self.assertNotIn(b"Images tested", response.data)
        self.assertNotIn(b"Export results", response.data)
        self.assertNotIn(b'id="validation-run-form"', response.data)

    def test_prediction_is_recorded_with_pending_reference_and_image(self):
        with patch.object(web, "predict", return_value=self._prediction()):
            response = self._submit_validation()

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Validation Results", response.data)
        self.assertIn(b"Reference disease", response.data)
        self.assertIn(b"Apple Scab", response.data)
        rows, total = validation_records.list_validations(db_path=validation_records.DB_PATH)
        self.assertEqual(total, 1)
        row = rows[0]
        self.assertEqual(row["verified_label"], "Apple___Apple_scab")
        self.assertEqual(row["match_label"], "Match")
        self.assertEqual(row["expert_review_status"], validation_records.REVIEW_PENDING)
        self.assertTrue(os.path.isfile(os.path.join(web.VALIDATION_IMAGES_DIR, row["stored_filename"])))
        image_response = self.client.get(f"/validation-images/{row['image_ref']}")
        self.assertEqual(image_response.status_code, 200)

    def test_selected_supported_class_is_compared_and_summary_updates(self):
        with patch.object(web, "predict", return_value=self._prediction()):
            response = self._submit_validation()
        self.assertEqual(response.status_code, 200)
        summary = validation_records.get_validation_summary(validation_records.DB_PATH)
        self.assertEqual(summary["images_tested"], 1)
        self.assertEqual(summary["correct_results"], 1)
        self.assertEqual(summary["incorrect_results"], 0)
        self.assertEqual(summary["test_accuracy_pct"], 100.0)

    def test_wrong_crop_reference_pair_is_rejected(self):
        payload = {
            "csrf_token": self.csrf,
            "crop_name": "Apple",
            "reference_label": "Tomato___Early_blight",
            "source_description": "Documented source",
            "reference_image": (io.BytesIO(self._image_bytes()), "leaf.jpg"),
        }
        with patch.object(web, "predict", return_value=self._prediction()) as mocked_predict:
            response = self.client.post("/validation/submit", data=payload, content_type="multipart/form-data", follow_redirects=True)
        mocked_predict.assert_not_called()
        self.assertIn(b"does not belong to the selected crop", response.data)
        _, total = validation_records.list_validations(db_path=validation_records.DB_PATH)
        self.assertEqual(total, 0)

    def test_reviewer_confirms_reference_then_records_separate_expert_review(self):
        with patch.object(web, "predict", return_value=self._prediction()):
            self._submit_validation()
        rows, _ = validation_records.list_validations(db_path=validation_records.DB_PATH)
        row = rows[0]

        login = self.client.post(
            "/validation/reviewer/login",
            data={"csrf_token": self.csrf, "reviewer_token": "review-token-test"},
            follow_redirects=True,
        )
        self.assertEqual(login.status_code, 200)
        verify = self.client.post(
            f"/validation/{row['id']}/verify",
            data={
                "csrf_token": self.csrf,
                "verified_label": "Apple___Apple_scab",
                "label_source": "Expert-confirmed reference label",
            },
            follow_redirects=True,
        )
        self.assertEqual(verify.status_code, 200)
        review = self.client.post(
            f"/validation/{row['id']}/review",
            data={
                "csrf_token": self.csrf,
                "expert_status": "Confirmed",
                "expert_diagnosis": "Apple scab",
                "expert_comments": "Visual reference reviewed",
                "reviewer_identity": "Agronomist A",
            },
            follow_redirects=True,
        )
        self.assertEqual(review.status_code, 200)
        summary = validation_records.get_validation_summary(validation_records.DB_PATH)
        self.assertEqual(summary["images_tested"], 1)
        self.assertEqual(summary["correct_results"], 1)
        self.assertEqual(summary["test_accuracy_pct"], 100.0)
        self.assertEqual(summary["expert_reviewed"], 1)

    def test_reviewer_changes_require_authorized_session_and_csv_is_available(self):
        with patch.object(web, "predict", return_value=self._prediction()):
            self._submit_validation()
        rows, _ = validation_records.list_validations(db_path=validation_records.DB_PATH)
        row = rows[0]
        forbidden = self.client.post(
            f"/validation/{row['id']}/verify",
            data={"csrf_token": self.csrf, "verified_label": "Apple___Apple_scab", "label_source": "source"},
        )
        self.assertEqual(forbidden.status_code, 403)
        export = self.client.get("/validation/export.csv")
        self.assertEqual(export.status_code, 200)
        self.assertIn(b"Verified Label", export.data)
        self.assertIn(b"Expert Review Status", export.data)


if __name__ == "__main__":
    unittest.main()

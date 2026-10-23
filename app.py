"""
CropCare AI — Flask web application.

Routes:
  GET  /            — Upload page (classify)
  POST /predict     — Run inference, save to DB, return result
  GET  /history     — Prediction history
  GET  /validation-results — Saved reference-image comparisons
  GET  /about       — Dataset, model, training documentation
  GET  /api/stats   — JSON stats endpoint
"""

import os
import sys
import uuid
import json
import time
import hmac
import secrets
import hashlib
from pathlib import Path

from flask import (Flask, render_template, request, redirect, url_for,
                   flash, jsonify, send_from_directory, session, abort, make_response)
from werkzeug.utils import secure_filename

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.settings import (
    SECRET_KEY, UPLOADS_DIR, ALLOWED_UPLOAD_EXTENSIONS,
    MAX_UPLOAD_SIZE_MB, TARGET_CLASSES, DISEASE_INFO,
    IDX_TO_CLASS, REPORTS_DIR, MODELS_DIR, VALIDATION_IMAGES_DIR,
)
from src.inference import predict, load_model, ImageValidationError
from src.database import init_db, save_prediction, get_all_predictions, get_stats
from src.validation_records import (
    add_validation_record,
    confirm_reference_label,
    get_validation_by_ref,
    get_validation_crops,
    init_validation_db,
    list_validations,
    record_expert_review,
    validation_csv,
)

app = Flask(__name__)
app.secret_key = os.getenv("CROPCARE_SECRET_KEY") or SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE_MB * 1024 * 1024
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = os.getenv('CROPCARE_HTTPS', '').lower() == 'true'

# Add enumerate to Jinja2 environment (for confusion matrix table)
app.jinja_env.globals['enumerate'] = enumerate

os.makedirs(UPLOADS_DIR, exist_ok=True)
init_db()
init_validation_db()
os.makedirs(VALIDATION_IMAGES_DIR, exist_ok=True)

PLANNED_REFERENCE_CASES = [
    {"id": "tomato_early_blight", "crop": "Tomato", "suggested_label": "Tomato___Early_blight", "title": "Tomato — Early blight"},
    {"id": "corn_common_rust", "crop": "Corn (Maize)", "suggested_label": "Corn_(maize)___Common_rust_", "title": "Corn — Common rust"},
    {"id": "potato_late_blight", "crop": "Potato", "suggested_label": "Potato___Late_blight", "title": "Potato — Late blight"},
    {"id": "apple_scab", "crop": "Apple", "suggested_label": "Apple___Apple_scab", "title": "Apple — Apple scab"},
    {"id": "tomato_healthy", "crop": "Tomato", "suggested_label": "Tomato___healthy", "title": "Tomato — Healthy leaf"},
]

# Pre-load model at startup (non-blocking — lazy on first request if fail)
_model_ready = False
_model_error = None

def _try_preload():
    global _model_ready, _model_error
    try:
        load_model()
        _model_ready = True
        print("[Flask] Model pre-loaded successfully.")
    except FileNotFoundError as e:
        _model_error = str(e)
        print(f"[Flask] Model not yet trained: {e}")
    except Exception as e:
        _model_error = str(e)
        print(f"[Flask] Model load error: {e}")

_try_preload()


# ──────────────────────────────────────────────────────
# Context processor — inject globals into all templates
# ──────────────────────────────────────────────────────

@app.context_processor
def inject_globals():
    return {
        "model_ready": _model_ready,
        "model_error": _model_error,
        "num_classes": len(TARGET_CLASSES),
        "max_upload_size_mb": MAX_UPLOAD_SIZE_MB,
        "allowed_upload_extensions": sorted(ext.lstrip(".") for ext in ALLOWED_UPLOAD_EXTENSIONS),
        "supported_classes": [
            {"class": cls,
             "display": DISEASE_INFO.get(cls, {}).get("display_name", cls),
             "crop": DISEASE_INFO.get(cls, {}).get("plant", ""),
             "healthy": DISEASE_INFO.get(cls, {}).get("is_healthy", False)}
            for cls in TARGET_CLASSES
        ],
        "supported_crops": sorted({DISEASE_INFO.get(cls, {}).get("plant", "Unknown") for cls in TARGET_CLASSES}),
    }


@app.before_request
def ensure_validation_csrf_token():
    session.setdefault("validation_csrf_token", secrets.token_urlsafe(32))


# ──────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────

def _validation_list_context():
    try:
        requested_page = max(1, int(request.args.get("validation_page", "1")))
    except ValueError:
        requested_page = 1
    validation_crop = request.args.get("validation_crop", "all")
    validation_status = request.args.get("validation_status", "all")
    validation_rows, validation_total = list_validations(
        crop=validation_crop,
        match_status=validation_status,
        page=requested_page,
    )
    return {
        "validation_records": validation_rows,
        "validation_total": validation_total,
        "validation_page": requested_page,
        "validation_pages": max(1, (validation_total + 19) // 20),
        "validation_crop": validation_crop,
        "validation_status": validation_status,
        "validation_crops": get_validation_crops(),
    }


@app.route("/")
def index():
    return render_template(
        "index.html",
        recent_predictions=get_all_predictions(limit=4),
        history_stats=get_stats(),
    )


@app.route("/validation-results")
def validation_results():
    return render_template("validation.html", **_validation_list_context())


def _validation_redirect():
    return redirect(url_for(
        "validation_results",
        validation_crop=request.form.get("validation_crop", "all"),
        validation_status=request.form.get("validation_status", "all"),
        validation_page=request.form.get("validation_page", "1"),
    ))


def _require_validation_csrf():
    expected = session.get("validation_csrf_token", "")
    submitted = request.form.get("csrf_token", "")
    if not expected or not hmac.compare_digest(expected, submitted):
        abort(400, description="Validation form expired. Reload the page and try again.")


def _reviewer_authenticated() -> bool:
    """Authenticate only sessions created with the currently configured reviewer token."""
    reviewer_token = os.getenv("CROPCARE_REVIEWER_TOKEN", "")
    secret_key = os.getenv("CROPCARE_SECRET_KEY", "")
    session_fingerprint = session.get("validation_reviewer_fingerprint", "")
    if not reviewer_token or not secret_key or not session_fingerprint:
        return False

    expected_fingerprint = hmac.new(
        secret_key.encode("utf-8"),
        reviewer_token.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(session_fingerprint, expected_fingerprint)


@app.route("/validation/submit", methods=["POST"])
def submit_validation_image():
    _require_validation_csrf()
    uploaded = request.files.get("reference_image")
    source = (request.form.get("source_description") or "").strip()
    submitted_crop = (request.form.get("crop_name") or "").strip()
    reference_label = (request.form.get("reference_label") or "").strip()
    if not uploaded or not uploaded.filename:
        flash("Choose a reference image to run the existing leaf prediction.", "error")
        return _validation_redirect()
    if not source:
        flash("Record where this reference image came from before submitting it.", "error")
        return _validation_redirect()
    if len(source) > 500:
        flash("Image source details must be 500 characters or fewer.", "error")
        return _validation_redirect()

    allowed_crops = set(DISEASE_INFO.get(class_name, {}).get("plant", "Unknown") for class_name in TARGET_CLASSES)
    if submitted_crop not in allowed_crops | {"Not supported by model"}:
        flash("Choose a crop from the supported list or mark it as not supported.", "error")
        return _validation_redirect()
    if reference_label == "__unsupported__":
        if submitted_crop != "Not supported by model":
            flash("Choose a supported crop when selecting a PlantVillage disease label.", "error")
            return _validation_redirect()
        confirmed_reference_label = f"Unsupported crop: {submitted_crop}"
    elif reference_label in TARGET_CLASSES:
        reference_crop = DISEASE_INFO.get(reference_label, {}).get("plant", "Unknown")
        if submitted_crop != reference_crop:
            flash("The selected reference disease does not belong to the selected crop.", "error")
            return _validation_redirect()
        confirmed_reference_label = reference_label
    else:
        flash("Choose a supported reference disease for the selected crop.", "error")
        return _validation_redirect()

    filename = secure_filename(uploaded.filename)
    image_bytes = uploaded.read()
    inference_result = predict(image_bytes, filename=filename, top_k=5)
    if not inference_result.get("success"):
        flash(f"Image was not recorded: {inference_result.get('error', 'prediction failed')}", "error")
        return _validation_redirect()

    image_ref = uuid.uuid4().hex
    extension = os.path.splitext(filename)[1].lower()
    stored_filename = f"{image_ref}{extension}"
    image_path = os.path.join(VALIDATION_IMAGES_DIR, stored_filename)
    try:
        os.makedirs(VALIDATION_IMAGES_DIR, exist_ok=True)
        with open(image_path, "wb") as image_file:
            image_file.write(image_bytes)
        crop = submitted_crop
        add_validation_record(
            image_ref=image_ref,
            stored_filename=stored_filename,
            original_filename=filename,
            crop=crop,
            predicted_class=inference_result["predicted_class"],
            predicted_display_name=inference_result["display_name"],
            prediction_confidence=inference_result["confidence"],
            source_description=source,
            verified_label=confirmed_reference_label,
            label_source=source,
        )
    except Exception:
        try:
            os.remove(image_path)
        except OSError:
            pass
        app.logger.exception("Could not save validation record")
        flash("The prediction completed, but the validation record could not be saved.", "error")
        return _validation_redirect()

    flash("Reference image tested. The match is based on the crop and disease label you selected.", "success")
    return _validation_redirect()


@app.route("/validation/reviewer/login", methods=["POST"])
def validation_reviewer_login():
    _require_validation_csrf()
    configured_token = os.getenv("CROPCARE_REVIEWER_TOKEN", "")
    if not os.getenv("CROPCARE_SECRET_KEY"):
        flash("Reviewer access requires CROPCARE_SECRET_KEY to be set in the app environment.", "error")
        return _validation_redirect()
    local_request = request.remote_addr in {"127.0.0.1", "::1"}
    if not request.is_secure and not local_request:
        flash("Reviewer sign-in requires HTTPS for non-local connections.", "error")
        return _validation_redirect()
    submitted_token = request.form.get("reviewer_token", "")
    if not configured_token:
        flash("Reviewer access is not configured. Set CROPCARE_REVIEWER_TOKEN and restart the app.", "error")
    elif hmac.compare_digest(submitted_token.encode("utf-8"), configured_token.encode("utf-8")):
        session["validation_reviewer_fingerprint"] = hmac.new(
            os.environ["CROPCARE_SECRET_KEY"].encode("utf-8"),
            configured_token.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        flash("Reviewer access enabled for this browser session.", "success")
    else:
        flash("Reviewer access was not granted. Check the review token.", "error")
    return _validation_redirect()


@app.route("/validation/reviewer/logout", methods=["POST"])
def validation_reviewer_logout():
    _require_validation_csrf()
    session.pop("validation_reviewer_fingerprint", None)
    flash("Reviewer session ended.", "info")
    return _validation_redirect()


@app.route("/validation/<int:record_id>/verify", methods=["POST"])
def verify_validation_label(record_id: int):
    _require_validation_csrf()
    if not _reviewer_authenticated():
        abort(403)
    verified_label = request.form.get("verified_label", "").strip()
    label_source = request.form.get("label_source", "").strip()
    try:
        confirm_reference_label(record_id, verified_label, label_source)
        flash("Reference label and source saved. Expert review remains separate.", "success")
    except (ValueError, LookupError) as exc:
        flash(str(exc), "error")
    return _validation_redirect()


@app.route("/validation/<int:record_id>/review", methods=["POST"])
def review_validation_record(record_id: int):
    _require_validation_csrf()
    if not _reviewer_authenticated():
        abort(403)
    try:
        record_expert_review(
            record_id,
            request.form.get("expert_status", ""),
            request.form.get("expert_diagnosis", ""),
            request.form.get("expert_comments", ""),
            request.form.get("reviewer_identity", ""),
        )
        flash("Expert review saved with reviewer and date.", "success")
    except (ValueError, LookupError) as exc:
        flash(str(exc), "error")
    return _validation_redirect()


@app.route("/validation-images/<image_ref>")
def validation_image(image_ref: str):
    record = get_validation_by_ref(image_ref)
    if not record:
        abort(404)
    return send_from_directory(VALIDATION_IMAGES_DIR, record["stored_filename"])


@app.route("/validation/export.csv")
def export_validation_csv():
    response = make_response(validation_csv())
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = "attachment; filename=cropcare_validation_results.csv"
    return response


@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        flash("No file selected.", "error")
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        flash("No file selected.", "error")
        return redirect(url_for("index"))

    # Read bytes (enforced by MAX_CONTENT_LENGTH)
    image_bytes = file.read()
    original_filename = secure_filename(file.filename)

    # Save uploaded file with unique name
    ext = os.path.splitext(original_filename)[1].lower()
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOADS_DIR, unique_name)
    with open(save_path, "wb") as f:
        f.write(image_bytes)

    # Run inference
    t0 = time.time()
    result = predict(image_bytes, filename=original_filename, top_k=5)
    inference_time_ms = round((time.time() - t0) * 1000, 1)

    if not result["success"]:
        flash(f"Prediction failed: {result['error']}", "error")
        return redirect(url_for("index"))

    result["inference_time_ms"] = inference_time_ms
    result["uploaded_filename"] = unique_name
    result["original_filename"] = original_filename
    result["crop"] = DISEASE_INFO.get(result["predicted_class"], {}).get("plant", "Crop not listed")

    # Save to database
    pred_id = save_prediction(original_filename, result)
    result["pred_id"] = pred_id

    return render_template("result.html", result=result)


@app.route("/history")
def history():
    predictions = get_all_predictions(limit=100)
    stats = get_stats()
    return render_template("history.html",
                           predictions=predictions, stats=stats)


@app.route("/about")
def about():
    # Load training report if available
    training_reports = {}
    mobile_report_path = os.path.join(REPORTS_DIR, "training_MobileNetV3_Large.json")
    if os.path.isfile(mobile_report_path):
        with open(mobile_report_path, encoding="utf-8") as f:
            training_reports["MobileNetV3_Large"] = json.load(f)
    for arch in ["CropCareNet_S", "CropCareNet_M", "CropCareNet_L"]:
        rpath = os.path.join(REPORTS_DIR, f"training_{arch}.json")
        if os.path.isfile(rpath):
            with open(rpath) as f:
                training_reports[arch] = json.load(f)

    eval_reports = {}
    mobile_eval_path = os.path.join(REPORTS_DIR, "eval_MobileNetV3_Large.json")
    if os.path.isfile(mobile_eval_path):
        with open(mobile_eval_path, encoding="utf-8") as f:
            eval_reports["MobileNetV3_Large"] = json.load(f)
    for arch in ["CropCareNet_S", "CropCareNet_M", "CropCareNet_L"]:
        epath = os.path.join(REPORTS_DIR, f"eval_{arch}.json")
        if os.path.isfile(epath):
            with open(epath) as f:
                eval_reports[arch] = json.load(f)

    audit_report = None
    audit_path = os.path.join(REPORTS_DIR, "dataset_audit.json")
    if os.path.isfile(audit_path):
        with open(audit_path) as f:
            audit_report = json.load(f)

    return render_template(
        "about.html",
        training_reports=training_reports,
        eval_reports=eval_reports,
        audit_report=audit_report,
        disease_info=DISEASE_INFO,
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOADS_DIR, filename)


@app.route("/api/stats")
def api_stats():
    return jsonify(get_stats())


@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "ok",
        "model_ready": _model_ready,
        "model_error": _model_error,
    })


# ──────────────────────────────────────────────────────
# Error handlers
# ──────────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    flash(f"File too large. Maximum size: {MAX_UPLOAD_SIZE_MB} MB.", "error")
    return redirect(url_for("index"))


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

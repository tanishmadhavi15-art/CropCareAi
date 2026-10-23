"""Central project configuration for the verified PlantVillage classes."""

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FULL_DATASET_MODELS_DIR = os.path.join(MODELS_DIR, "full_dataset")
MOBILENET_MODELS_DIR = os.path.join(MODELS_DIR, "mobilenet_v3")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "static", "uploads")
VALIDATION_IMAGES_DIR = os.path.join(UPLOADS_DIR, "validation")
DB_PATH = os.path.join(PROJECT_ROOT, "cropcare.db")

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pth")
TRAINING_LOG_PATH = os.path.join(REPORTS_DIR, "training_log.json")
AUDIT_REPORT_PATH = os.path.join(REPORTS_DIR, "dataset_audit.json")

# ──────────────────────────────────────────────
# Dataset — all populated, readable PlantVillage class directories
# ──────────────────────────────────────────────
TARGET_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]
_CROP_FILTER = [c.strip() for c in os.environ.get("CROPCARE_CROPS", "").split(",") if c.strip()]
if _CROP_FILTER:
    TARGET_CLASSES = [c for c in TARGET_CLASSES if c.split("___")[0] in _CROP_FILTER]

NUM_CLASSES = len(TARGET_CLASSES)
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(TARGET_CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(TARGET_CLASSES)}

# ──────────────────────────────────────────────
# Image pre-processing
# ──────────────────────────────────────────────
IMG_SIZE = 128          # default; 128x128 for efficient training and inference
MOBILENET_IMG_SIZE = 224
MOBILENET_BATCH_SIZE = 8
MOBILENET_EPOCHS = 20
MOBILENET_HEAD_EPOCHS = 2
MOBILENET_HEAD_LR = 1e-3
MOBILENET_FINETUNE_LR = 1e-4
IMG_CHANNELS = 3        # RGB
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

# ──────────────────────────────────────────────
# Training hyper-parameters
# ──────────────────────────────────────────────
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15       # remaining 0.15 is test
RANDOM_SEED = 42
NUM_WORKERS = 0         # 0 recommended on Windows to prevent IPC/spawn issues
EARLY_STOP_PATIENCE = 7 # epochs without val-F1 improvement

# ──────────────────────────────────────────────
# Flask / web
# ──────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = 10
ALLOWED_UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SECRET_KEY = "cropcare-dev-secret-change-in-production"

# ──────────────────────────────────────────────
# Disease metadata — symptoms, plant group, & management
# ──────────────────────────────────────────────
DISEASE_INFO = {
    # ── Apple ──
    "Apple___Apple_scab": {
        "plant": "Apple",
        "display_name": "Apple Scab",
        "is_healthy": False,
        "symptoms": "Olive-green to dark velvety spots on leaves, becoming cracked and scabby over time.",
        "management": "Apply preventive captan or myclobutanil fungicides. Prune canopy for good sunlight and rake fallen leaves.",
    },
    "Apple___Black_rot": {
        "plant": "Apple",
        "display_name": "Apple Black Rot",
        "is_healthy": False,
        "symptoms": "Circular brown leaf spots with purple margins (frog-eye leaf spots); dark mummified fruit.",
        "management": "Prune out dead wood, cankers, and mummified fruit. Apply sulfur or copper fungicides during early bud break.",
    },
    "Apple___Cedar_apple_rust": {
        "plant": "Apple",
        "display_name": "Cedar Apple Rust",
        "is_healthy": False,
        "symptoms": "Bright yellow-orange circular lesions on upper leaf surface with tube-like spore structures beneath.",
        "management": "Remove nearby red cedar/juniper hosts within 1-2 miles if possible. Apply myclobutanil fungicides at pink bud stage.",
    },
    "Apple___healthy": {
        "plant": "Apple",
        "display_name": "Apple Healthy",
        "is_healthy": True,
        "symptoms": "Vibrant green leaves free from blemishes, spots, or abnormal curling.",
        "management": "Maintain balanced seasonal pruning, proper fertilization, and monitor periodically for pests.",
    },

    # ── Corn (Maize) ──
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "plant": "Corn (Maize)",
        "display_name": "Corn Gray Leaf Spot",
        "is_healthy": False,
        "symptoms": "Rectangular tan-to-gray lesions bounded by leaf veins running parallel to leaf edges.",
        "management": "Plant resistant hybrids, practice minimum 1-year crop rotation, and apply strobilurin or triazole fungicides if severe.",
    },
    "Corn_(maize)___Common_rust_": {
        "plant": "Corn (Maize)",
        "display_name": "Corn Common Rust",
        "is_healthy": False,
        "symptoms": "Oval, cinnamon-brown powdery pustules on both upper and lower leaf surfaces.",
        "management": "Utilize rust-resistant corn varieties. Fungicide treatment is rarely needed unless infection appears prior to tasseling.",
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "plant": "Corn (Maize)",
        "display_name": "Northern Corn Leaf Blight",
        "is_healthy": False,
        "symptoms": "Long, elliptical grayish-green or tan cigar-shaped lesions (1-6 inches long).",
        "management": "Plant resistant hybrids with Ht gene resistance. Bury residue with tillage and rotate away from continuous corn.",
    },
    "Corn_(maize)___healthy": {
        "plant": "Corn (Maize)",
        "display_name": "Corn Healthy",
        "is_healthy": True,
        "symptoms": "Uniform deep green leaf blades without lesions, pustules, or striping.",
        "management": "Provide adequate nitrogen fertilization, appropriate planting density, and sufficient soil moisture.",
    },

    # ── Grape ──
    "Grape___Black_rot": {
        "plant": "Grape",
        "display_name": "Grape Black Rot",
        "is_healthy": False,
        "symptoms": "Reddish-brown circular leaf spots with tiny black pycnidia pimples; berries shrivel into black mummies.",
        "management": "Maintain open trellis canopy for rapid leaf drying. Apply mancozeb or captan fungicides from early shoot growth through veraison.",
    },
    "Grape___Esca_(Black_Measles)": {
        "plant": "Grape",
        "display_name": "Grape Esca (Black Measles)",
        "is_healthy": False,
        "symptoms": "Tiger-stripe patterns of chlorotic and necrotic tissue between leaf veins; dark spots on berries.",
        "management": "Protect pruning wounds with wound sealants. Avoid severe pruning during wet weather. Remove dying cordons.",
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "plant": "Grape",
        "display_name": "Grape Leaf Blight",
        "is_healthy": False,
        "symptoms": "Irregularly shaped dark brown spots on foliage, leading to premature leaf scorch and defoliation.",
        "management": "Apply copper or chlorothalonil sprays after harvest or early spring. Prune vines to improve aeration.",
    },
    "Grape___healthy": {
        "plant": "Grape",
        "display_name": "Grape Healthy",
        "is_healthy": True,
        "symptoms": "Broad, vibrant green palmately lobed leaves with healthy venation and vigorous cane growth.",
        "management": "Ensure balanced shoot thinning, canopy management, and regular irrigation during fruit set.",
    },

    # ── Bell Pepper ──
    "Pepper,_bell___Bacterial_spot": {
        "plant": "Bell Pepper",
        "display_name": "Pepper Bacterial Spot",
        "is_healthy": False,
        "symptoms": "Small, water-soaked blistering spots on leaves that turn brown with yellow margins; leaf drop.",
        "management": "Use certified disease-free seed. Apply fixed copper sprays combined with mancozeb. Avoid handling wet plants.",
    },
    "Pepper,_bell___healthy": {
        "plant": "Bell Pepper",
        "display_name": "Pepper Healthy",
        "is_healthy": True,
        "symptoms": "Deep glossy green leaves with erect stems and normal flower/fruit initiation.",
        "management": "Maintain consistent moisture to prevent blossom end rot, mulch beds, and rotate crops annually.",
    },

    # ── Potato ──
    "Potato___Early_blight": {
        "plant": "Potato",
        "display_name": "Potato Early Blight",
        "is_healthy": False,
        "symptoms": "Dark brown circular spots with concentric target-ring patterns primarily on older foliage.",
        "management": "Ensure plant vigor with adequate fertility. Apply protectant fungicides (chlorothalonil, mancozeb). Avoid overhead watering.",
    },
    "Potato___Late_blight": {
        "plant": "Potato",
        "display_name": "Potato Late Blight",
        "is_healthy": False,
        "symptoms": "Water-soaked irregular dark spots rapidly expanding; white mildew fringe on undersides during wet weather.",
        "management": "Destroy volunteer potatoes and infected tubers. Apply systemic fungicides immediately upon local blight warnings.",
    },
    "Potato___healthy": {
        "plant": "Potato",
        "display_name": "Potato Healthy",
        "is_healthy": True,
        "symptoms": "Lush compound green leaves free from chlorosis, necrotic margins, or wilting.",
        "management": "Hill soil properly around developing tubers and practice 3-year rotation away from nightshades.",
    },

    # ── Tomato ──
    "Tomato___Bacterial_spot": {
        "plant": "Tomato",
        "display_name": "Tomato Bacterial Spot",
        "is_healthy": False,
        "symptoms": "Small, dark water-soaked spots with yellow halos; severe defoliation during rainy conditions.",
        "management": "Use certified clean seed. Apply copper bactericides preventively. Avoid overhead irrigation.",
    },
    "Tomato___Early_blight": {
        "plant": "Tomato",
        "display_name": "Tomato Early Blight",
        "is_healthy": False,
        "symptoms": "Concentric dark brown rings forming target-shaped lesions on mature lower leaves.",
        "management": "Mulch to prevent soil splash. Prune bottom leaves. Apply copper or chlorothalonil fungicides.",
    },
    "Tomato___Late_blight": {
        "plant": "Tomato",
        "display_name": "Tomato Late Blight",
        "is_healthy": False,
        "symptoms": "Large, greasy dark lesions with white fungal fuzz on leaf undersides in cool, damp conditions.",
        "management": "Remove and destroy infected foliage immediately. Apply protective fungicides before wet periods.",
    },
    "Tomato___Leaf_Mold": {
        "plant": "Tomato",
        "display_name": "Tomato Leaf Mold",
        "is_healthy": False,
        "symptoms": "Pale greenish-yellow blotches on upper leaf surfaces, olive-green velvety mold underneath.",
        "management": "Improve greenhouse ventilation. Reduce relative humidity below 85%. Avoid wetting leaves.",
    },
    "Tomato___Septoria_leaf_spot": {
        "plant": "Tomato",
        "display_name": "Tomato Septoria Leaf Spot",
        "is_healthy": False,
        "symptoms": "Numerous circular 2-3 mm spots with gray centers and dark brown borders with tiny black dots inside.",
        "management": "Remove lower infected foliage. Stake plants for airflow. Apply chlorothalonil or copper fungicides.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "plant": "Tomato",
        "display_name": "Tomato Spider Mites",
        "is_healthy": False,
        "symptoms": "Fine yellow stippling on leaf surface, followed by bronze discoloration and fine webbing under leaves.",
        "management": "Apply insecticidal soaps or neem oil. Release predatory mites (Phytoseiulus persimilis) in greenhouses.",
    },
    "Tomato___Target_Spot": {
        "plant": "Tomato",
        "display_name": "Tomato Target Spot",
        "is_healthy": False,
        "symptoms": "Small pinpoint brown lesions that expand into circular spots with light brown centers and dark rings.",
        "management": "Ensure adequate plant spacing. Remove crop residues after harvest. Apply preventive fungicides.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "plant": "Tomato",
        "display_name": "Tomato Yellow Leaf Curl Virus",
        "is_healthy": False,
        "symptoms": "Upward curling and severe yellowing of leaf margins; stunted bushy plant growth with flower drop.",
        "management": "Control whitefly vectors using yellow sticky traps and insect screens. Use virus-resistant cultivars.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "plant": "Tomato",
        "display_name": "Tomato Mosaic Virus",
        "is_healthy": False,
        "symptoms": "Mottled light and dark green mosaic patterns on leaves, leaf distortion, and fern-like appearance.",
        "management": "Wash hands and disinfect tools thoroughly (tobacco mosaic virus is mechanically transmitted). Plant resistant varieties.",
    },
    "Tomato___healthy": {
        "plant": "Tomato",
        "display_name": "Tomato Healthy",
        "is_healthy": True,
        "symptoms": "Uniform deep green foliage with balanced vegetative growth, vigorous blossom set, and healthy stems.",
        "management": "Provide balanced watering, mulch, cage/stake plants, and monitor regularly for early symptoms.",
    },
}

# Use transparent, conservative guidance for observed classes without curated
# entries. The canonical label is kept intact; no disease-specific symptoms or
# treatments are inferred from the label alone.
for _class_name in TARGET_CLASSES:
    if _class_name in DISEASE_INFO:
        continue
    _plant, _condition = _class_name.split("___", 1)
    _display_condition = _condition.replace("_", " ").strip().title()
    DISEASE_INFO[_class_name] = {
        "plant": _plant.replace("_", " ").replace(",", "").title(),
        "display_name": f"{_plant.replace('_', ' ').replace(',', '').title()} {_display_condition}",
        "is_healthy": _condition.lower() == "healthy",
        "symptoms": (
            f"This image was classified as '{_class_name}' based on PlantVillage labels. "
            "Symptoms can vary; confirm the result with a qualified local specialist."
        ),
        "management": (
            "Do not apply treatment based only on an image prediction. Confirm the diagnosis "
            "with a qualified agricultural extension service before taking action."
        ),
    }

import os
import re
import sqlite3

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)


app = Flask(__name__)
app.secret_key = os.environ.get(
    "PLANTCARE_SECRET_KEY",
    "plantcare-development-secret-key"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(BASE_DIR, "instance", "users.db")

DEVICE = torch.device("cpu")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

LEAF_CLASSES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

FRUIT_CLASSES = [
    "Tomato__Healthy",
    "Tomato__Rotten"
]


def find_model_path(filename):
    for folder in ("model", "models"):
        path = os.path.join(BASE_DIR, folder, filename)
        if os.path.isfile(path):
            return path
    return os.path.join(BASE_DIR, "model", filename)


LEAF_MODEL_PATH = find_model_path("tomato_disease_model.pth")
FRUIT_MODEL_PATH = find_model_path("fruit_disease_model.pth")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def create_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )
    return model


def load_model(model_path, num_classes):
    if not os.path.isfile(model_path):
        print("MODEL FILE NOT FOUND:", model_path)
        return None

    try:
        checkpoint = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True
        )

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint.state_dict()

        cleaned_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[7:]
            cleaned_state_dict[key] = value

        model = create_model(num_classes)
        model.load_state_dict(cleaned_state_dict, strict=True)
        model.to(DEVICE)
        model.eval()

        print("MODEL LOADED:", model_path)
        return model

    except Exception as error:
        print("MODEL LOADING FAILED:", model_path)
        print("Reason:", error)
        return None


leaf_model = load_model(
    LEAF_MODEL_PATH,
    len(LEAF_CLASSES)
)

fruit_model = load_model(
    FRUIT_MODEL_PATH,
    len(FRUIT_CLASSES)
)


LEAF_CARE = {
    "Tomato_Bacterial_spot": (
        "Bacterial Spot",
        "Disease symptoms detected.",
        [
            "Remove badly affected leaves and dispose of them safely.",
            "Avoid overhead watering and working with wet plants.",
            "Improve airflow and avoid handling plants when wet.",
            "Ask a local agricultural expert about suitable disease management."
        ]
    ),
    "Tomato_Early_blight": (
        "Early Blight",
        "Disease symptoms detected.",
        [
            "Remove severely affected lower leaves.",
            "Water near the soil instead of wetting the foliage.",
            "Use mulch to reduce soil splashing onto leaves.",
            "Consider a locally recommended fungicide according to its label."
        ]
    ),
    "Tomato_Late_blight": (
        "Late Blight",
        "Potentially serious disease symptoms detected.",
        [
            "Isolate affected plants where practical.",
            "Avoid overhead irrigation and prolonged leaf wetness.",
            "Remove affected plant material safely.",
            "Seek prompt local agricultural advice for disease management."
        ]
    ),
    "Tomato_Leaf_Mold": (
        "Leaf Mold",
        "Disease symptoms detected.",
        [
            "Improve ventilation around plants.",
            "Avoid wetting leaves when watering.",
            "Reduce excessive humidity where possible.",
            "Ask an agricultural expert about suitable treatment if symptoms spread."
        ]
    ),
    "Tomato_Septoria_leaf_spot": (
        "Septoria Leaf Spot",
        "Disease symptoms detected.",
        [
            "Remove affected lower leaves when practical.",
            "Keep foliage dry and avoid overhead watering.",
            "Remove plant debris and weeds around tomato plants.",
            "Use only locally approved treatments according to their labels."
        ]
    ),
    "Tomato_Spider_mites_Two_spotted_spider_mite": (
        "Two-Spotted Spider Mites",
        "Possible spider mite damage detected.",
        [
            "Inspect leaf undersides for mites and fine webbing.",
            "Reduce dust and water-stress around plants.",
            "Use an appropriate pest-control method recommended locally.",
            "Follow the product label and protect beneficial insects."
        ]
    ),
    "Tomato__Target_Spot": (
        "Target Spot",
        "Disease symptoms detected.",
        [
            "Remove severely affected leaves.",
            "Improve airflow and avoid wet foliage.",
            "Clear infected plant debris.",
            "Seek local advice if the disease continues to spread."
        ]
    ),
    "Tomato__Tomato_YellowLeaf__Curl_Virus": (
        "Tomato Yellow Leaf Curl Virus",
        "Possible viral disease symptoms detected.",
        [
            "Check plants for whiteflies, which can spread this virus.",
            "Remove severely affected plants when advised by a local expert.",
            "Control weeds around the growing area.",
            "Use locally recommended whitefly management and resistant varieties."
        ]
    ),
    "Tomato__Tomato_mosaic_virus": (
        "Tomato Mosaic Virus",
        "Possible viral disease symptoms detected.",
        [
            "Avoid touching healthy plants after handling affected plants.",
            "Clean tools and wash hands between plants.",
            "Remove affected plants when advised by an agricultural expert.",
            "Use clean seed and avoid tobacco contamination on hands."
        ]
    ),
    "Tomato_healthy": (
        "Healthy Leaf",
        "No disease was identified among the model's supported classes.",
        [
            "Continue appropriate watering and regular plant inspection.",
            "Provide adequate sunlight and airflow.",
            "Monitor leaves regularly for new symptoms."
        ]
    )
}


FRUIT_CARE = {
    "Tomato__Healthy": (
        "Healthy Fruit",
        "The fruit was classified as healthy.",
        [
            "Store tomatoes in a clean, dry place.",
            "Handle fruit gently to avoid bruising.",
            "Inspect stored fruit regularly."
        ]
    ),
    "Tomato__Rotten": (
        "Rotten Fruit",
        "Possible fruit rot or spoilage detected.",
        [
            "Separate visibly rotten fruit from healthy tomatoes.",
            "Discard spoiled fruit safely; do not eat fruit that appears rotten.",
            "Keep storage containers clean and dry.",
            "Avoid bruising fruit and inspect nearby tomatoes for spoilage."
        ]
    )
}


def predict_image(image, model, classes):
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)

    predicted_index = predicted.item()
    confidence_value = float(confidence.item() * 100)
    prediction = classes[predicted_index]

    return prediction, confidence_value


def build_result(upload, model, classes, care_dictionary, label):
    if not upload or not upload.filename:
        return None, None

    extension = os.path.splitext(upload.filename)[1].lower()

    if extension not in IMAGE_EXTENSIONS:
        return None, f"Please upload a valid {label} image (JPG, PNG, WEBP, or BMP)."

    if model is None:
        return None, (
            f"The {label} model could not be loaded. "
            "Check the model file and the VS Code terminal."
        )

    try:
        with Image.open(upload.stream) as source_image:
            image = source_image.convert("RGB")
            prediction, confidence = predict_image(
                image,
                model,
                classes
            )

        display_name, summary, suggestions = care_dictionary[prediction]

        result = {
            "label": label,
            "prediction": prediction,
            "display_name": display_name,
            "summary": summary,
            "confidence": round(confidence, 2),
            "low_confidence": confidence < 70,
            "suggestions": suggestions,
            "is_healthy": prediction in (
                "Tomato_healthy",
                "Tomato__Healthy"
            )
        }

        return result, None

    except (UnidentifiedImageError, OSError):
        return None, f"The {label} file is not a readable image. Please choose another."
    except Exception as error:
        print(f"{label} prediction error:", error)
        return None, f"Could not analyze the {label} image. Check the terminal for details."


def init_db():
    os.makedirs(os.path.dirname(DATABASE), exist_ok=True)

    with sqlite3.connect(DATABASE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        """)


init_db()


def valid_gmail(email):
    pattern = r"^[a-zA-Z0-9._%+-]+@gmail\.com$"
    return re.fullmatch(pattern, email) is not None


@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not password:
            return render_template(
                "login.html",
                error="Please enter email and password."
            )

        if not valid_gmail(email):
            return render_template(
                "login.html",
                error="Please enter a valid Gmail address."
            )

        with sqlite3.connect(DATABASE) as conn:
            user = conn.execute(
                """
                SELECT id, name, email, password
                FROM users
                WHERE email = ?
                """,
                (email,)
            ).fetchone()

        if not user:
            return render_template(
                "login.html",
                error="Account not found. Please sign up first."
            )

        if not check_password_hash(user[3], password):
            return render_template(
                "login.html",
                error="Incorrect password."
            )

        session["logged_in"] = True
        session["user_id"] = user[0]
        session["username"] = user[1]
        session["email"] = user[2]

        return redirect(url_for("home"))

    return render_template(
        "login.html",
        success=request.args.get("success")
    )


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not all([name, email, password, confirm_password]):
            return render_template(
                "signup.html",
                error="Please fill in all fields."
            )

        if not valid_gmail(email):
            return render_template(
                "signup.html",
                error="Please enter a valid Gmail address."
            )

        if len(password) < 6:
            return render_template(
                "signup.html",
                error="Password must contain at least 6 characters."
            )

        if password != confirm_password:
            return render_template(
                "signup.html",
                error="Passwords do not match."
            )

        try:
            with sqlite3.connect(DATABASE) as conn:
                conn.execute(
                    """
                    INSERT INTO users (name, email, password)
                    VALUES (?, ?, ?)
                    """,
                    (
                        name,
                        email,
                        generate_password_hash(password)
                    )
                )
        except sqlite3.IntegrityError:
            return render_template(
                "signup.html",
                error="This email is already registered. Please login."
            )

        return redirect(
            url_for(
                "login",
                success="Account created successfully. Please login."
            )
        )

    return render_template("signup.html")


@app.route("/home")
def home():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    return render_template(
        "home.html",
        username=session.get("username")
    )


@app.route("/detection", methods=["GET", "POST"])
def detection():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    leaf_result = None
    fruit_result = None
    leaf_error = None
    fruit_error = None
    error = None

    if request.method == "POST":
        leaf_upload = request.files.get("leaf_image")
        fruit_upload = request.files.get("fruit_image")

        leaf_selected = bool(
            leaf_upload and leaf_upload.filename
        )
        fruit_selected = bool(
            fruit_upload and fruit_upload.filename
        )

        if not leaf_selected and not fruit_selected:
            error = "Please select a leaf image, a fruit image, or both."
        else:
            if leaf_selected:
                leaf_result, leaf_error = build_result(
                    leaf_upload,
                    leaf_model,
                    LEAF_CLASSES,
                    LEAF_CARE,
                    "leaf"
                )

            if fruit_selected:
                fruit_result, fruit_error = build_result(
                    fruit_upload,
                    fruit_model,
                    FRUIT_CLASSES,
                    FRUIT_CARE,
                    "fruit"
                )

    return render_template(
        "detection.html",
        leaf_result=leaf_result,
        fruit_result=fruit_result,
        leaf_error=leaf_error,
        fruit_error=fruit_error,
        error=error
    )


@app.route("/about")
def about():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    return render_template("about.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)
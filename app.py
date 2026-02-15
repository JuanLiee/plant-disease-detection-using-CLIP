from flask import Flask, render_template, request, redirect, session, url_for, flash
from werkzeug.utils import secure_filename
from model_users import db, User
from model import predict_image, DISEASE_TREATMENTS
from ollama_ai import ask_ollama
import os

# ================= CONFIG =================
app = Flask(__name__)
app.secret_key = "supersecretkey"  # nanti deploy ganti env var

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

CONFIDENCE_THRESHOLD = 0.15
MAX_RESULTS = 3

db.init_app(app)
with app.app_context():
    db.create_all()

# ================= HELPERS =================
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def get_current_user():
    if "user_id" in session:
        return User.query.get(session["user_id"])
    return None

# ================= ROUTES LOGIN & REGISTER =================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        if User.query.filter_by(email=email).first():
            return "Email already registered"

        if User.query.filter_by(username=username).first():
            return "Username already taken"

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        return redirect("/login")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            session["user_id"] = user.id
            return redirect("/")
        return "Login failed"

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user_id", None)
    return redirect("/login")


@app.route("/settings", methods=["GET", "POST"])
def settings():
    user = get_current_user()
    if not user:
        return redirect("/login")

    if request.method == "POST":
        user.preferred_theme = request.form.get("theme", "light")
        user.notifications = request.form.get("notifications") is not None
        db.session.commit()
        flash("Settings saved ✅", "success")
        return redirect("/settings")

    return render_template("settings.html", user=user)

# ================= ROUTES AI =================
@app.route("/", methods=["GET", "POST"])
def index():
    user = get_current_user()
    if not user:
        return redirect("/login")

    result = []
    explanation = ""
    image_url = ""
    treatment = {"chemical": [], "organic": [], "prevention": []}
    warning = ""
    error = ""

    if request.method == "POST":
        if "image" not in request.files:
            error = "No image uploaded"
            return render_template("index.html", error=error, user=user)

        file = request.files["image"]

        if file.filename == "":
            error = "No image selected"
            return render_template("index.html", error=error, user=user)

        if not allowed_file(file.filename):
            error = "Invalid file type. Only png/jpg/jpeg allowed."
            return render_template("index.html", error=error, user=user)

        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        # ✅ biar bisa dipakai di <img src="">
        image_url = url_for("static", filename=f"uploads/{filename}")

        # ===== AI PREDICTION =====
        try:
            predictions = predict_image(path)
        except Exception as e:
            print("Predict failed:", e)
            error = "Model failed to analyze image"
            return render_template("index.html", error=error, user=user)

        result = [(label, round(conf * 100, 2)) for label, conf in predictions[:MAX_RESULTS]]
        top_label, top_conf = predictions[0]

        # ===== OLLAMA EXPLANATION + FALLBACK (buat cloud) =====
        try:
            explanation = ask_ollama(top_label)
        except Exception as e:
            print("Ollama failed:", e)
            explanation = (
                f"Disease Overview:\n{top_label}\n\n"
                "Possible Causes:\n- Pathogen infection (fungal/bacterial)\n- High humidity / poor airflow\n\n"
                "Common Symptoms:\n- Spots, discoloration, curling\n- Leaf yellowing or mold\n\n"
                "Suggested Solutions:\n- Improve airflow and reduce leaf wetness\n"
                "- Remove infected leaves\n"
                "- Consider suitable fungicide if needed\n"
            )

        # ===== LOOKUP OBAT & SOLUSI =====
        treatment = DISEASE_TREATMENTS.get(top_label, {"chemical": [], "organic": [], "prevention": []})

        # ===== WARNING =====
        if top_conf < CONFIDENCE_THRESHOLD:
            warning = "⚠️ Prediction confidence is low. Results may be inaccurate."

    return render_template(
        "index.html",
        result=result,
        explanation=explanation,
        treatment=treatment,
        image_url=image_url,
        warning=warning,
        error=error,
        user=user,
        max_results=MAX_RESULTS,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )

# ================= MAIN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


from flask import Flask, render_template, request, redirect, session, url_for, flash
from werkzeug.utils import secure_filename
from model_users import db, User
from model import predict_image, DISEASE_TREATMENTS
from ollama_ai import ask_ollama
import os

# ================= CONFIG =================
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "supersecretkey")

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
    user = get_current_user()

    if request.method == "POST":
        username = request.form["username"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        if User.query.filter_by(email=email).first():
            flash("Email already registered", "error")
            return redirect("/register")

        if User.query.filter_by(username=username).first():
            flash("Username already taken", "error")
            return redirect("/register")

        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        flash("Register success ✅ Please login.", "success")
        return redirect("/login")

    return render_template("register.html", user=user)

@app.route("/login", methods=["GET", "POST"])
def login():
    user = get_current_user()

    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        u = User.query.filter_by(email=email).first()
        if u and u.check_password(password):
            session["user_id"] = u.id
            flash("Welcome back ✅", "success")
            return redirect("/")

        flash("Email / password wrong ❌", "error")
        return redirect("/login")

    return render_template("login.html", user=user)

@app.route("/logout")
def logout():
    session.pop("user_id", None)
    flash("Logged out 👋", "success")
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

        # ===== AI EXPLANATION (Ollama if available, otherwise fallback) =====
        try:
            explanation = ask_ollama(top_label)
            if not explanation or not explanation.strip():
                raise RuntimeError("Empty Ollama output")
        except Exception as e:
            print("Ollama failed:", e)

            t = DISEASE_TREATMENTS.get(top_label, {"chemical": [], "organic": [], "prevention": []})
            chem = "\n".join([f"- {x['name']}" for x in t.get("chemical", [])]) or "- (none)"
            org = "\n".join([f"- {x['name']}" for x in t.get("organic", [])]) or "- (none)"
            prev = "\n".join([f"- {x}" for x in t.get("prevention", [])]) or "- (none)"

            explanation = (
                f"Disease Overview:\n"
                f"{top_label} ({round(top_conf*100, 2)}% confidence)\n\n"
                f"Possible Causes:\n"
                f"- Pathogens (fungal/bacterial) affecting leaves\n"
                f"- High humidity / poor airflow\n"
                f"- Water splashing onto foliage\n\n"
                f"Common Symptoms:\n"
                f"- Spots, discoloration, curling, or mold\n"
                f"- Yellowing or wilting in affected areas\n\n"
                f"Suggested Solutions:\n"
                f"Organic:\n{org}\n\n"
                f"Chemical:\n{chem}\n\n"
                f"Prevention Tips:\n{prev}\n"
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
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

# ================= MAIN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

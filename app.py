"""
DeepFake Detector — Flask Web App
Run: python app.py
Visit: http://localhost:5000
"""

import os
import io
import sys
import base64
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import DeepFakeDetector

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT = os.environ.get("CHECKPOINT_PATH", "outputs/best_model.pth")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
MAX_BYTES  = 10 * 1024 * 1024   # 10 MB

# ── Load model once at startup ────────────────────────────────────────────────
model = None

def load_model():
    global model
    if not os.path.exists(CHECKPOINT):
        print(f"[WARN] Checkpoint not found at '{CHECKPOINT}'. "
              "Train first: python src/train.py")
        return
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    m = DeepFakeDetector(dropout_rate=ckpt.get("config", {}).get("dropout_rate", 0.4))
    m.load_state_dict(ckpt["model_state_dict"])
    m.to(DEVICE).eval()
    model = m
    print(f"[OK] Model loaded from '{CHECKPOINT}' on {DEVICE}")

# ── Preprocessing ─────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run 'python src/train.py' first."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    raw  = file.read()
    if len(raw) > MAX_BYTES:
        return jsonify({"error": "File too large (max 10 MB)."}), 413

    try:
        img    = Image.open(io.BytesIO(raw)).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(DEVICE)
    except Exception as e:
        return jsonify({"error": f"Could not process image: {e}"}), 400

    with torch.no_grad():
        logit = model(tensor)
        prob  = torch.sigmoid(logit).item()   # probability of FAKE

    fake_pct = round(prob * 100, 1)
    real_pct = round((1 - prob) * 100, 1)
    verdict  = "FAKE" if prob > 0.5 else "REAL"

    if prob > 0.85 or prob < 0.15:
        confidence = "High"
    elif prob > 0.70 or prob < 0.30:
        confidence = "Medium"
    else:
        confidence = "Low"

    # Return small thumbnail as base64 for display
    thumb = img.copy()
    thumb.thumbnail((400, 400))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=85)
    thumb_b64 = base64.b64encode(buf.getvalue()).decode()

    return jsonify({
        "verdict":      verdict,
        "fake_prob":    fake_pct,
        "real_prob":    real_pct,
        "confidence":   confidence,
        "thumbnail":    f"data:image/jpeg;base64,{thumb_b64}",
    })


@app.route("/health")
def health():
    return jsonify({"model_loaded": model is not None, "device": str(DEVICE)})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    app.run(debug=True, host="0.0.0.0", port=5000)

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import xgboost as xgb
import pickle
from utils.preprocess import preprocess_orbscan_image
from utils.features import extract_cnn_features, extract_classical_features
from utils.safety import munnerlyn_ablation, compute_rsb, evaluate_safety
from utils.decision import final_decision

app = FastAPI(title="Refractive Surgery Decision API", version="1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
kc_model = xgb.XGBClassifier()
kc_model.load_model("models/kc_xgboost_model.json")

surgery_model = xgb.XGBClassifier()
surgery_model.load_model("models/surgery_xgboost_model.json")

with open("models/surgery_label_encoder.pkl", "rb") as f:
    le = pickle.load(f)


@app.post("/predict")
async def predict_surgery(
    anterior: UploadFile = File(...),
    axial: UploadFile = File(...),
    posterior: UploadFile = File(...),
    pachy: UploadFile = File(...),
    age: float = Form(...),
    astig_value_D: float = Form(...),
    astig_axis_deg: float = Form(...),
    kmax_value_D: float = Form(...),
    pachy_central_um: float = Form(...),
    asphericity_anterior: float = Form(...),
    asphericity_posterior: float = Form(...),
    diopters: float = Form(3.0),
    oz_mm: float = Form(6.0)
):
    # ---------- 1. Preprocess all images ----------
    def process(file):
        bytes_data = await file.read()
        img = preprocess_orbscan_image(bytes_data)
        return img, extract_cnn_features(img), extract_classical_features(img)

    maps = {}
    for name, file in [
        ("anterior", anterior),
        ("axial", axial),
        ("posterior", posterior),
        ("pachy", pachy),
    ]:
        maps[name] = await process(file)

    # CNN + Classical fusion
    cnn_vec = np.concatenate([maps[m][1] for m in maps])
    classical_vec = np.concatenate([maps[m][2] for m in maps])

    clinical_vec = np.array([
        age,
        astig_value_D,
        astig_axis_deg,
        kmax_value_D,
        pachy_central_um,
        asphericity_anterior,
        asphericity_posterior
    ], dtype=np.float32)

    full_vec = np.concatenate([cnn_vec, classical_vec, clinical_vec])

    # ---------- 2. KC prediction ----------
    kc_prob = float(kc_model.predict_proba([full_vec])[0][1])

    # ---------- 3. Surgery ML prediction ----------
    s_prob = surgery_model.predict_proba([full_vec])[0]
    s_pred = le.inverse_transform([np.argmax(s_prob)])[0]

    # ---------- 4. Safety calculations ----------
    ablation = munnerlyn_ablation(diopters, oz_mm)
    rsb = compute_rsb(pachy_central_um, ablation)
    safety = evaluate_safety(pachy_central_um, kmax_value_D, kc_prob, rsb)

    # ---------- 5. Final decision (override ML if unsafe) ----------
    final_surgery, reason = final_decision(safety, s_pred, s_prob)

    # ---------- 6. Response ----------
    return {
        "KC_probability": kc_prob,
        "RSB_um": rsb,
        "ablation_depth_um": ablation,
        "safety": safety,
        "ml_surgery_prediction": s_pred,
        "surgery_probabilities": dict(zip(le.classes_, s_prob.tolist())),
        "final_recommended_surgery": final_surgery,
        "override_reason": reason
    }

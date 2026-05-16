"""
Workwise - API REST v2
Recibe variables ya calculadas desde Spring Boot y devuelve la predicción.

POST /predict
{
  "experiencia_candidato": 3,
  "cumple_experiencia":    1,
  "brecha_experiencia":    1,
  "nivel_candidato":       4,
  "cumple_nivel":          1,
  "match_habilidades":     0.75,
  "match_categoria":       1
}
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from tensorflow import keras
import joblib
import json
import os

MODEL_DIR = "saved_model"

app = FastAPI(
    title="Workwise ML API v2",
    description="Predice aceptación laboral comparando candidato vs oferta.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model      = None
preprocessor_scaler = None
metadata   = None

# Columnas en el orden exacto que espera el modelo
SCALE_COLS      = ["experiencia_candidato", "brecha_experiencia", "nivel_candidato"]
PASSTHROUGH_COLS = ["cumple_experiencia", "cumple_nivel", "match_habilidades", "match_categoria"]


@app.on_event("startup")
def load_model():
    global model, preprocessor_scaler, metadata

    if not os.path.exists(MODEL_DIR):
        print(f"⚠ Directorio '{MODEL_DIR}' no encontrado. Ejecuta model.py primero.")
        return

    model               = keras.models.load_model(f"{MODEL_DIR}/model.h5")
    preprocessor_scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")

    with open(f"{MODEL_DIR}/metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("✓ Modelo v2 cargado correctamente.")


# ─── Schema ──────────────────────────────────────────────────────────────────

class PrediccionInput(BaseModel):
    experiencia_candidato: int   = Field(..., ge=0, le=50, example=3)
    cumple_experiencia:    int   = Field(..., ge=0, le=1,  example=1)
    brecha_experiencia:    int   = Field(..., example=1)
    nivel_candidato:       int   = Field(..., ge=0, le=6,  example=4)
    cumple_nivel:          int   = Field(..., ge=0, le=1,  example=1)
    match_habilidades:     float = Field(..., ge=0.0, le=1.0, example=0.75)
    match_categoria:       int   = Field(..., ge=0, le=1,  example=1)


class PrediccionResponse(BaseModel):
    aceptado:     bool
    probabilidad: float
    confianza:    str
    mensaje:      str
    detalle: dict   # desglose para que Spring Boot lo muestre si quiere


# ─── Lógica ───────────────────────────────────────────────────────────────────

def _preprocess(data: dict) -> np.ndarray:
    df     = pd.DataFrame([data])
    scaled = preprocessor_scaler.transform(df[SCALE_COLS].values)
    passthrough = df[PASSTHROUGH_COLS].values
    return np.hstack([scaled, passthrough])


def _confianza(prob: float) -> str:
    if prob >= 0.75 or prob <= 0.25:
        return "Alta"
    if prob >= 0.60 or prob <= 0.40:
        return "Media"
    return "Baja"


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "version": "2.0"}


@app.post("/predict", response_model=PrediccionResponse)
def predict(data: PrediccionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")

    try:
        X    = _preprocess(data.model_dump())
        prob = float(model.predict(X, verbose=0)[0][0])
        aceptado = prob >= 0.5

        return PrediccionResponse(
            aceptado=aceptado,
            probabilidad=round(prob, 4),
            confianza=_confianza(prob),
            mensaje=(
                "El candidato tiene alta compatibilidad con esta oferta."
                if aceptado else
                "El candidato tiene baja compatibilidad con esta oferta."
            ),
            detalle={
                "cumple_experiencia": bool(data.cumple_experiencia),
                "cumple_nivel_estudio": bool(data.cumple_nivel),
                "match_habilidades_pct": round(data.match_habilidades * 100),
                "match_categoria": bool(data.match_categoria),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
"""
Workwise - API REST v3
Recibe las 8 variables cuantitativas desde Spring Boot y devuelve la predicción.

POST /predict
{
  "habilidades_oferta":    6,
  "habilidades_match":     3,
  "experiencia_oferta":    2,
  "experiencia_candidato": 4,
  "nivel_oferta":          4,
  "nivel_candidato":       4,
  "sector_oferta":         0,
  "sector_candidato":      0
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

# Orden exacto en que el scaler fue entrenado
SCALE_COLS = [
    "habilidades_oferta",
    "habilidades_match",
    "experiencia_oferta",
    "experiencia_candidato",
    "nivel_oferta",
    "nivel_candidato",
    "sector_oferta",
    "sector_candidato",
]

app = FastAPI(
    title="Workwise ML API v3",
    description="Predicción de compatibilidad laboral con 8 variables cuantitativas.",
    version="3.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model  = None
scaler = None
meta   = None


@app.on_event("startup")
def load_model():
    global model, scaler, meta

    if not os.path.exists(MODEL_DIR):
        print(f"⚠ Directorio '{MODEL_DIR}' no encontrado. Ejecuta model.py primero.")
        return

    model  = keras.models.load_model(f"{MODEL_DIR}/model.h5")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")

    with open(f"{MODEL_DIR}/metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    print("✓ Modelo v3 cargado correctamente.")


# ─── Schemas ──────────────────────────────────────────────────────────────────

class PrediccionInput(BaseModel):
    habilidades_oferta:    int   = Field(..., ge=0,   le=20,  example=6)
    habilidades_match:     int   = Field(..., ge=0,   le=20,  example=3)
    experiencia_oferta:    int   = Field(..., ge=0,   le=30,  example=2)
    experiencia_candidato: int   = Field(..., ge=0,   le=50,  example=4)
    nivel_oferta:          int   = Field(..., ge=0,   le=6,   example=4)
    nivel_candidato:       int   = Field(..., ge=0,   le=6,   example=4)
    sector_oferta:         int   = Field(..., ge=0,   le=11,  example=0)
    sector_candidato:      int   = Field(..., ge=0,   le=11,  example=0)


class PrediccionResponse(BaseModel):
    aceptado:     bool
    probabilidad: float
    confianza:    str
    mensaje:      str
    detalle:      dict


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _preprocess(data: dict) -> np.ndarray:
    df = pd.DataFrame([data])
    return scaler.transform(df[SCALE_COLS].values)


def _confianza(prob: float) -> str:
    if prob >= 0.75 or prob <= 0.25:
        return "Alta"
    if prob >= 0.60 or prob <= 0.40:
        return "Media"
    return "Baja"


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "version": "3.0"}


@app.post("/predict", response_model=PrediccionResponse)
def predict(data: PrediccionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")

    # Validación de negocio: habilidades_match no puede superar habilidades_oferta
    if data.habilidades_match > data.habilidades_oferta:
        raise HTTPException(
            status_code=422,
            detail="habilidades_match no puede ser mayor que habilidades_oferta."
        )

    try:
        X    = _preprocess(data.model_dump())
        prob = float(model.predict(X, verbose=0)[0][0])
        aceptado = prob >= 0.5

        # Detalle legible para el frontend
        hab_oferta = data.habilidades_oferta
        hab_match  = data.habilidades_match
        pct_hab    = round((hab_match / hab_oferta * 100) if hab_oferta > 0 else 0)

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
                "habilidades_oferta":      hab_oferta,
                "habilidades_match":       hab_match,
                "habilidades_pct":         pct_hab,
                "cumple_experiencia":      data.experiencia_candidato >= data.experiencia_oferta,
                "brecha_experiencia":      data.experiencia_candidato - data.experiencia_oferta,
                "cumple_nivel":            data.nivel_candidato >= data.nivel_oferta,
                "match_sector":            data.sector_candidato == data.sector_oferta,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
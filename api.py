"""
Workwise - API REST para predicción laboral
Consumible desde Spring Boot (Java) y Flutter (Dart/Dio)

Endpoints:
  POST /predict        → predicción individual
  POST /predict/batch  → predicción en lote
  GET  /health         → estado del servicio
  GET  /metadata       → variables y valores aceptados
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
import json
import os

# ─── Config ────────────────────────────────────────────────────────────────────

MODEL_DIR = "saved_model"

TIPO_EMPLEO    = ["Tiempo_Completo", "Medio_Tiempo", "Por_Horas", "Freelance"]
MODALIDAD      = ["Presencial", "Remoto", "Hibrido"]
TIPO_CONTRATO  = ["Indefinido", "Practicas", "Obra_Labor", "Fijo"]
NIVEL_ESTUDIO  = [
    "Sin_estudios", "Bachiller", "Tecnico_Tecnologo",
    "Tecnologo_Universitario", "Universitario", "Master", "Doctorado"
]
SECTOR = [
    "Aeroespacial", "Agricultura", "Agroindustria", "Ambiental", "ArtesCreativas",
    "Automotriz", "BienesRaices", "Biotecnologia", "Comercio", "Construccion",
    "Consultoria", "Deportes", "Diseno", "Educacion", "Energia", "Finanzas",
    "Gobierno", "Investigacion", "Legal", "Logistica", "Manufactura", "Marketing",
    "Medios", "Mineria", "Naval", "Quimica", "RRHH", "Salud", "Seguros",
    "Servicios", "Social", "Tecnologia", "Textil", "Transporte", "Turismo"
]

# ─── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Workwise ML API",
    description="Red neuronal para predecir la aceptación de una persona en una oferta laboral.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # En producción, limitar a tu dominio
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Estado global del modelo ───────────────────────────────────────────────────

model = None
encoders = None
scaler = None
metadata = None


@app.on_event("startup")
def load_model():
    global model, encoders, scaler, metadata

    if not os.path.exists(MODEL_DIR):
        print(f"⚠ Directorio '{MODEL_DIR}' no encontrado. Ejecuta model.py primero.")
        return

    model    = keras.models.load_model(f"{MODEL_DIR}/model.h5")
    encoders = joblib.load(f"{MODEL_DIR}/encoders.pkl")
    scaler   = joblib.load(f"{MODEL_DIR}/scaler.pkl")

    with open(f"{MODEL_DIR}/metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("✓ Modelo cargado correctamente.")


# ─── Schemas ────────────────────────────────────────────────────────────────────

class PostulanteInput(BaseModel):
    tipo_empleo:   str = Field(..., example="Tiempo_Completo")
    modalidad:     str = Field(..., example="Remoto")
    tipo_contrato: str = Field(..., example="Indefinido")
    experiencia:   int = Field(..., ge=0, le=50, example=3)
    nivel_estudio: str = Field(..., example="Universitario")
    sector:        str = Field(..., example="Tecnologia")

    @field_validator("tipo_empleo")
    @classmethod
    def val_tipo_empleo(cls, v):
        if v not in TIPO_EMPLEO:
            raise ValueError(f"tipo_empleo debe ser uno de: {TIPO_EMPLEO}")
        return v

    @field_validator("modalidad")
    @classmethod
    def val_modalidad(cls, v):
        if v not in MODALIDAD:
            raise ValueError(f"modalidad debe ser uno de: {MODALIDAD}")
        return v

    @field_validator("tipo_contrato")
    @classmethod
    def val_tipo_contrato(cls, v):
        if v not in TIPO_CONTRATO:
            raise ValueError(f"tipo_contrato debe ser uno de: {TIPO_CONTRATO}")
        return v

    @field_validator("nivel_estudio")
    @classmethod
    def val_nivel_estudio(cls, v):
        if v not in NIVEL_ESTUDIO:
            raise ValueError(f"nivel_estudio debe ser uno de: {NIVEL_ESTUDIO}")
        return v

    @field_validator("sector")
    @classmethod
    def val_sector(cls, v):
        if v not in SECTOR:
            raise ValueError(f"sector debe ser uno de: {SECTOR}")
        return v


class PrediccionResponse(BaseModel):
    aceptado:    bool
    probabilidad: float = Field(..., description="Probabilidad de aceptación (0.0 - 1.0)")
    confianza:   str   = Field(..., description="Alta / Media / Baja")
    mensaje:     str


class BatchRequest(BaseModel):
    postulantes: list[PostulanteInput]


class BatchResponse(BaseModel):
    resultados: list[PrediccionResponse]
    total:      int
    aceptados:  int


# ─── Lógica de predicción ───────────────────────────────────────────────────────

def _preprocess(data: dict | list[dict]) -> np.ndarray:
    """Preprocesa uno o varios postulantes."""
    df = pd.DataFrame(data if isinstance(data, list) else [data])

    cat_cols = ["tipo_empleo", "modalidad", "tipo_contrato", "nivel_estudio", "sector"]
    for col in cat_cols:
        df[col] = encoders[col].transform(df[col])

    num_scaled = scaler.transform(df[["experiencia"]].values)
    X = np.hstack([df[cat_cols].values, num_scaled])
    return X


def _confianza(prob: float) -> str:
    if prob >= 0.75 or prob <= 0.25:
        return "Alta"
    if prob >= 0.60 or prob <= 0.40:
        return "Media"
    return "Baja"


def _build_response(prob: float) -> PrediccionResponse:
    aceptado = bool(prob >= 0.5)
    return PrediccionResponse(
        aceptado=aceptado,
        probabilidad=round(float(prob), 4),
        confianza=_confianza(prob),
        mensaje=(
            "El postulante tiene alta probabilidad de ser aceptado." if aceptado
            else "El postulante tiene baja probabilidad de ser aceptado."
        )
    )


# ─── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Sistema"])
def health():
    """Verifica que el servicio y el modelo estén operativos."""
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@app.get("/metadata", tags=["Sistema"])
def get_metadata():
    """Devuelve las variables y valores válidos para las predicciones."""
    return {
        "variables": {
            "tipo_empleo":   TIPO_EMPLEO,
            "modalidad":     MODALIDAD,
            "tipo_contrato": TIPO_CONTRATO,
            "nivel_estudio": NIVEL_ESTUDIO,
            "sector":        SECTOR,
            "experiencia":   "Entero entre 0 y 50 (años)"
        }
    }


@app.post("/predict", response_model=PrediccionResponse, tags=["Predicción"])
def predict(postulante: PostulanteInput):
    """
    Predice si un postulante será aceptado en una oferta laboral.

    **Ejemplo de uso desde Java (Spring Boot):**
    ```java
    RestTemplate rt = new RestTemplate();
    ResponseEntity<Map> resp = rt.postForEntity(
        "http://localhost:8000/predict", body, Map.class);
    ```

    **Ejemplo de uso desde Flutter/Dart (Dio):**
    ```dart
    final response = await dio.post('/predict', data: postulante.toJson());
    ```
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible. Contacta al administrador.")

    try:
        X = _preprocess(postulante.model_dump())
        prob = model.predict(X, verbose=0)[0][0]
        return _build_response(prob)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")


@app.post("/predict/batch", response_model=BatchResponse, tags=["Predicción"])
def predict_batch(request: BatchRequest):
    """
    Predice en lote para múltiples postulantes a la vez.
    Útil para evaluar una lista de candidatos de una oferta.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")

    if len(request.postulantes) > 100:
        raise HTTPException(status_code=400, detail="Máximo 100 postulantes por lote.")

    try:
        data = [p.model_dump() for p in request.postulantes]
        X = _preprocess(data)
        probs = model.predict(X, verbose=0).flatten()

        resultados = [_build_response(p) for p in probs]
        aceptados  = sum(1 for r in resultados if r.aceptado)

        return BatchResponse(
            resultados=resultados,
            total=len(resultados),
            aceptados=aceptados
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción batch: {str(e)}")


# ─── Main ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

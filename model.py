"""
Workwise - Red Neuronal de Predicción de Aceptación Laboral v3
Modelo: Clasificación binaria (Aceptado / No Aceptado)

Variables de entrada (8) — todas cuantitativas/numéricas:
  1. habilidades_oferta     → cuántas habilidades pide la oferta (entero)
  2. habilidades_match      → cuántas de esas tiene el candidato (entero)
  3. experiencia_oferta     → años mínimos requeridos por la oferta
  4. experiencia_candidato  → años del candidato (formulario)
  5. nivel_oferta           → nivel educativo requerido (0-6)
  6. nivel_candidato        → nivel educativo del candidato (0-6, formulario)
  7. sector_oferta          → sector de la oferta (codificado 0-N)
  8. sector_candidato       → sector del candidato (formulario, codificado 0-N)

La red aprende por sí sola cuánto pesa cada diferencia.
No se calculan ratios ni variables derivadas antes de pasar al modelo.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import json
import os

# ─── Constantes ────────────────────────────────────────────────────────────────

NIVEL_ESTUDIO = [
    "Sin_estudios",            # 0
    "Bachiller",               # 1
    "Tecnico_Tecnologo",       # 2
    "Tecnologo_Universitario", # 3
    "Universitario",           # 4
    "Master",                  # 5
    "Doctorado",               # 6
]
NIVEL_PESO = {n: i for i, n in enumerate(NIVEL_ESTUDIO)}

# Sectores — deben coincidir con los valores reales que llegan desde Spring Boot
SECTORES = [
    "Tecnología",      # 0
    "Salud",           # 1
    "Educación",       # 2
    "Finanzas",        # 3
    "Construcción",    # 4
    "Marketing",       # 5
    "Logística",       # 6
    "E-commerce",      # 7
    "Turismo",         # 8
    "Industria",       # 9
    "Otro",            # 10
]
SECTOR_PESO = {s: i for i, s in enumerate(SECTORES)}
N_SECTORES  = len(SECTORES)

MODEL_DIR = "saved_model"

# Columnas que se escalan con StandardScaler
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


# ─── Generación de datos sintéticos ────────────────────────────────────────────

def generate_synthetic_data(n_samples: int = 12000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)

    # ── Oferta ───────────────────────────────────────────────────────────────
    habilidades_oferta    = np.random.randint(1, 8, n_samples)      # 1-7 habilidades
    experiencia_oferta    = np.random.randint(0, 11, n_samples)     # 0-10 años
    nivel_oferta          = np.random.randint(0, 7,  n_samples)     # 0-6
    sector_oferta         = np.random.randint(0, N_SECTORES, n_samples)

    # ── Candidato ────────────────────────────────────────────────────────────
    # habilidades_match: entre 0 y lo que pide la oferta
    habilidades_match     = np.array([
        np.random.randint(0, habilidades_oferta[i] + 1)
        for i in range(n_samples)
    ])
    experiencia_candidato = np.random.randint(0, 21, n_samples)     # 0-20 años
    nivel_candidato       = np.random.randint(0, 7,  n_samples)     # 0-6
    sector_candidato      = np.random.randint(0, N_SECTORES, n_samples)

    # ── Probabilidad de aceptación (reglas de negocio) ────────────────────────
    prob = np.zeros(n_samples)
    for i in range(n_samples):
        p = 0.10  # base mínima

        # --- Habilidades: ratio de coincidencia ponderado ---
        ratio_hab = habilidades_match[i] / habilidades_oferta[i]
        p += ratio_hab * 0.30          # hasta +0.30 si tiene todas

        # --- Experiencia ---
        if experiencia_candidato[i] >= experiencia_oferta[i]:
            p += 0.22
            exceso = min((experiencia_candidato[i] - experiencia_oferta[i]) / 10, 1.0)
            p += exceso * 0.08         # bonus por superar el mínimo
        else:
            deficit = experiencia_oferta[i] - experiencia_candidato[i]
            p -= min(deficit / 5, 0.18)

        # --- Nivel educativo ---
        if nivel_candidato[i] >= nivel_oferta[i]:
            p += 0.18
            brecha_nivel = nivel_candidato[i] - nivel_oferta[i]
            p += min(brecha_nivel / 6, 1.0) * 0.05
        else:
            p -= 0.10

        # --- Sector ---
        if sector_candidato[i] == sector_oferta[i]:
            p += 0.12

        # Ruido
        prob[i] = float(np.clip(p + np.random.normal(0, 0.06), 0.0, 1.0))

    df = pd.DataFrame({
        "habilidades_oferta":    habilidades_oferta,
        "habilidades_match":     habilidades_match,
        "experiencia_oferta":    experiencia_oferta,
        "experiencia_candidato": experiencia_candidato,
        "nivel_oferta":          nivel_oferta,
        "nivel_candidato":       nivel_candidato,
        "sector_oferta":         sector_oferta,
        "sector_candidato":      sector_candidato,
        "aceptado":              (prob > 0.50).astype(int),
    })

    print(f"✓ Dataset generado: {n_samples} muestras")
    print(f"  Aceptados:    {df['aceptado'].sum()} ({df['aceptado'].mean()*100:.1f}%)")
    print(f"  No aceptados: {(df['aceptado']==0).sum()} ({(1-df['aceptado'].mean())*100:.1f}%)")
    return df


# ─── Preprocesador ─────────────────────────────────────────────────────────────

class WorkwisePreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.scaler.fit_transform(df[SCALE_COLS].values)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(df[SCALE_COLS].values)

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.scaler, f"{directory}/scaler.pkl")
        print(f"✓ Preprocesador guardado en '{directory}/'")

    def load(self, directory: str):
        self.scaler = joblib.load(f"{directory}/scaler.pkl")
        print(f"✓ Preprocesador cargado desde '{directory}/'")


# ─── Red Neuronal ───────────────────────────────────────────────────────────────

def build_model(input_dim: int) -> keras.Model:
    """
    8 entradas → 64 → 32 → 16 → 1 (sigmoid)
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(32, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.15),

        layers.Dense(16, activation="relu"),
        layers.Dropout(0.1),

        layers.Dense(1, activation="sigmoid"),
    ], name="workwise_predictor_v3")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


# ─── Entrenamiento ──────────────────────────────────────────────────────────────

def train():
    print("\n══════════════════════════════════════════════")
    print("   WORKWISE v3 — Entrenamiento del Modelo")
    print("══════════════════════════════════════════════\n")

    df = generate_synthetic_data(n_samples=12000)

    preprocessor = WorkwisePreprocessor()
    X = preprocessor.fit_transform(df[SCALE_COLS])
    y = df["aceptado"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Train: {len(X_train)} | Test: {len(X_test)}")

    model = build_model(input_dim=X.shape[1])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=5, min_lr=1e-6, verbose=1,
        ),
    ]

    print("\n▶ Entrenando...\n")
    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=1,
    )

    print("\n── Evaluación en Test ──────────────────────────")
    y_pred_prob = model.predict(X_test).flatten()
    y_pred      = (y_pred_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["No Aceptado", "Aceptado"]))
    print(confusion_matrix(y_test, y_pred))

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(f"{MODEL_DIR}/model.h5", include_optimizer=False)
    preprocessor.save(MODEL_DIR)

    metadata = {
        "version":        "3.0",
        "input_features": SCALE_COLS,
        "output":         "aceptado (0=No, 1=Si)",
        "threshold":      0.5,
        "accuracy_test":  round(float(acc), 4),
        "nivel_peso":     NIVEL_PESO,
        "sector_peso":    SECTOR_PESO,
    }
    with open(f"{MODEL_DIR}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Modelo v3 guardado en '{MODEL_DIR}/'")
    print("══════════════════════════════════════════════\n")


if __name__ == "__main__":
    train()
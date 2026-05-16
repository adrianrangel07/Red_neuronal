"""
Workwise - Red Neuronal de Predicción de Aceptación Laboral v2
Modelo: Clasificación binaria (Aceptado / No Aceptado)

Variables de entrada (7):
  1. experiencia_candidato     → años que ingresa el usuario
  2. cumple_experiencia        → 1 si experiencia_candidato >= experiencia_oferta
  3. brecha_experiencia        → experiencia_candidato - experiencia_oferta (puede ser negativa)
  4. nivel_estudio_candidato   → nivel que selecciona el usuario (codificado)
  5. cumple_nivel              → 1 si nivel_candidato >= nivel_oferta
  6. match_habilidades         → % de habilidades de la oferta que tiene el candidato (0.0-1.0)
  7. match_categoria           → 1 si la categoría del candidato coincide con la de la oferta
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import json
import os

# ─── Constantes ────────────────────────────────────────────────────────────────

NIVEL_ESTUDIO = [
    "Sin_estudios",           # 0 — menor nivel
    "Bachiller",              # 1
    "Tecnico_Tecnologo",      # 2
    "Tecnologo_Universitario",# 3
    "Universitario",          # 4
    "Master",                 # 5
    "Doctorado"               # 6 — mayor nivel
]

# Mapa numérico para comparar niveles
NIVEL_PESO = {n: i for i, n in enumerate(NIVEL_ESTUDIO)}

CATEGORIAS = [
    "Ingenieria", "Desarrollo", "Salud", "Administracion",
    "Diseno", "Otros", "Tecnologia", "Construccion"
]

MODEL_DIR = "saved_model"


# ─── Generación de datos sintéticos ────────────────────────────────────────────

def generate_synthetic_data(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """
    Genera datos sintéticos con variables que comparan candidato vs oferta.
    Las reglas de negocio reflejan criterios reales de selección laboral.
    """
    np.random.seed(random_state)

    # ── Datos del candidato ──────────────────────────────────────────────────
    experiencia_candidato = np.random.randint(0, 21, n_samples)
    nivel_candidato_idx   = np.random.randint(0, len(NIVEL_ESTUDIO), n_samples)
    categoria_candidato   = np.random.randint(0, len(CATEGORIAS), n_samples)

    # ── Datos de la oferta ───────────────────────────────────────────────────
    experiencia_oferta    = np.random.randint(0, 11, n_samples)
    nivel_oferta_idx      = np.random.randint(0, len(NIVEL_ESTUDIO), n_samples)
    categoria_oferta      = np.random.randint(0, len(CATEGORIAS), n_samples)
    habilidades_oferta    = np.random.randint(1, 6, n_samples)   # 1-5 habilidades requeridas

    # Habilidades que tiene el candidato de las requeridas (0 a todas)
    habilidades_comunes   = np.array([
        np.random.randint(0, habilidades_oferta[i] + 1)
        for i in range(n_samples)
    ])

    # ── Variables derivadas (lo que la red realmente aprende) ────────────────
    cumple_experiencia  = (experiencia_candidato >= experiencia_oferta).astype(int)
    brecha_experiencia  = experiencia_candidato - experiencia_oferta
    cumple_nivel        = (nivel_candidato_idx >= nivel_oferta_idx).astype(int)
    match_habilidades   = habilidades_comunes / habilidades_oferta          # 0.0 - 1.0
    match_categoria     = (categoria_candidato == categoria_oferta).astype(int)

    # ── Calcular probabilidad de aceptación ──────────────────────────────────
    prob = np.zeros(n_samples)

    for i in range(n_samples):
        p = 0.15  # base mínima

        # Cumplir experiencia mínima es el factor más importante
        if cumple_experiencia[i]:
            p += 0.25
            # Bonus por superar el mínimo con creces
            extra = min(brecha_experiencia[i] / 10, 1.0)
            p += extra * 0.10
        else:
            # Penalización por no llegar al mínimo
            deficit = abs(brecha_experiencia[i])
            p -= min(deficit / 5, 0.15)

        # Cumplir nivel de estudio requerido
        if cumple_nivel[i]:
            p += 0.20
            # Bonus por superar el nivel requerido
            brecha_nivel = nivel_candidato_idx[i] - nivel_oferta_idx[i]
            p += min(brecha_nivel / 6, 1.0) * 0.05
        else:
            p -= 0.10

        # Match de habilidades — muy influyente
        p += match_habilidades[i] * 0.25

        # Match de categoría profesional
        if match_categoria[i]:
            p += 0.15

        # Ruido realista
        prob[i] = np.clip(p + np.random.normal(0, 0.06), 0, 1)

    df = pd.DataFrame({
        "experiencia_candidato": experiencia_candidato,
        "cumple_experiencia":    cumple_experiencia,
        "brecha_experiencia":    brecha_experiencia,
        "nivel_candidato":       nivel_candidato_idx,
        "cumple_nivel":          cumple_nivel,
        "match_habilidades":     match_habilidades,
        "match_categoria":       match_categoria,
        "aceptado":              (prob > 0.50).astype(int)
    })

    print(f"✓ Dataset generado: {n_samples} muestras")
    print(f"  Aceptados:    {df['aceptado'].sum()} ({df['aceptado'].mean()*100:.1f}%)")
    print(f"  No aceptados: {(df['aceptado']==0).sum()} ({(1-df['aceptado'].mean())*100:.1f}%)")
    return df


# ─── Preprocesamiento ───────────────────────────────────────────────────────────

class WorkwisePreprocessor:
    """
    Escala las variables numéricas continuas.
    Las variables binarias (cumple_*) y match_habilidades ya están en 0-1,
    por lo que solo escalamos experiencia, brecha y nivel.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.scale_cols  = ["experiencia_candidato", "brecha_experiencia", "nivel_candidato"]
        self.passthrough = ["cumple_experiencia", "cumple_nivel", "match_habilidades", "match_categoria"]

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        scaled = self.scaler.fit_transform(df[self.scale_cols].values)
        passthrough = df[self.passthrough].values
        return np.hstack([scaled, passthrough])

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        scaled = self.scaler.transform(df[self.scale_cols].values)
        passthrough = df[self.passthrough].values
        return np.hstack([scaled, passthrough])

    def transform_single(self, row: dict) -> np.ndarray:
        """Transforma un único registro (para la API)."""
        df = pd.DataFrame([row])
        return self.transform(df)

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
    Red neuronal para clasificación binaria.
    Arquitectura: 7 entradas → 64 → 32 → 16 → 1 (sigmoide)
    Más simple que v1 porque las variables ya son semánticas (no categóricas crudas).
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

        layers.Dense(1, activation="sigmoid")
    ], name="workwise_predictor_v2")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    return model


# ─── Entrenamiento ──────────────────────────────────────────────────────────────

def train():
    print("\n══════════════════════════════════════════════")
    print("   WORKWISE v2 — Entrenamiento del Modelo")
    print("══════════════════════════════════════════════\n")

    # 1. Generar datos
    df = generate_synthetic_data(n_samples=10000)

    # 2. Preprocesar
    feature_cols = [
        "experiencia_candidato", "cumple_experiencia", "brecha_experiencia",
        "nivel_candidato", "cumple_nivel", "match_habilidades", "match_categoria"
    ]
    preprocessor = WorkwisePreprocessor()
    X = preprocessor.fit_transform(df[feature_cols])
    y = df["aceptado"].values

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Train: {len(X_train)} | Test: {len(X_test)}")

    # 4. Modelo
    model = build_model(input_dim=X.shape[1])
    model.summary()

    # 5. Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=5, min_lr=1e-6, verbose=1
        )
    ]

    # 6. Entrenar
    print("\n▶ Entrenando...\n")
    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=1
    )

    # 7. Evaluar
    print("\n── Evaluación en Test ──────────────────────────")
    y_pred_prob = model.predict(X_test).flatten()
    y_pred      = (y_pred_prob > 0.5).astype(int)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=["No Aceptado", "Aceptado"]))
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    # 8. Guardar
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(f"{MODEL_DIR}/model.h5", include_optimizer=False)
    preprocessor.save(MODEL_DIR)

    metadata = {
        "version": "2.0",
        "input_features": feature_cols,
        "output": "aceptado (0 = No, 1 = Si)",
        "threshold": 0.5,
        "accuracy_test": round(float(accuracy_score(y_test, y_pred)), 4),
        "nivel_estudio_map": NIVEL_PESO,
        "categorias": CATEGORIAS
    }
    with open(f"{MODEL_DIR}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Modelo v2 guardado en '{MODEL_DIR}/'")
    print("══════════════════════════════════════════════\n")


if __name__ == "__main__":
    train()
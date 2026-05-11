"""
Workwise - Red Neuronal de Predicción de Aceptación Laboral
Modelo: Clasificación binaria (Aceptado / No Aceptado)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import json
import os

# ─── Constantes ────────────────────────────────────────────────────────────────

TIPO_EMPLEO = ["Tiempo_Completo", "Medio_Tiempo", "Por_Horas", "Freelance"]

MODALIDAD = ["Presencial", "Remoto", "Hibrido"]

TIPO_CONTRATO = ["Indefinido", "Practicas", "Obra_Labor", "Fijo"]

NIVEL_ESTUDIO = [
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

MODEL_DIR = "saved_model"


# ─── Generación de datos sintéticos ────────────────────────────────────────────

def generate_synthetic_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Genera un dataset sintético con reglas de negocio realistas para
    simular postulaciones laborales en el contexto colombiano.
    """
    np.random.seed(random_state)

    data = {
        "tipo_empleo":   np.random.choice(TIPO_EMPLEO, n_samples),
        "modalidad":     np.random.choice(MODALIDAD, n_samples),
        "tipo_contrato": np.random.choice(TIPO_CONTRATO, n_samples),
        "experiencia":   np.random.randint(0, 21, n_samples),          # 0-20 años
        "nivel_estudio": np.random.choice(NIVEL_ESTUDIO, n_samples),
        "sector":        np.random.choice(SECTOR, n_samples),
    }
    df = pd.DataFrame(data)

    # Mapa de peso por nivel de estudio
    estudio_peso = {
        "Sin_estudios": 0, "Bachiller": 1, "Tecnico_Tecnologo": 2,
        "Tecnologo_Universitario": 3, "Universitario": 4,
        "Master": 5, "Doctorado": 6
    }

    # Sectores de alta demanda (mayor probabilidad de aceptación)
    sectores_alta_demanda = {
        "Tecnologia", "Salud", "Finanzas", "Logistica",
        "Manufactura", "Educacion", "Energia"
    }

    # Calcular probabilidad de aceptación con reglas de negocio
    prob = np.zeros(n_samples)

    for i, row in df.iterrows():
        p = 0.3  # base

        # Experiencia (mayor experiencia = más probabilidad)
        p += min(row["experiencia"] / 20, 1.0) * 0.25

        # Nivel de estudio
        p += (estudio_peso[row["nivel_estudio"]] / 6) * 0.25

        # Sector de alta demanda
        if row["sector"] in sectores_alta_demanda:
            p += 0.10

        # Tipo de empleo (Tiempo completo más solicitado)
        if row["tipo_empleo"] == "Tiempo_Completo":
            p += 0.05

        # Tipo de contrato (Indefinido más atractivo)
        if row["tipo_contrato"] == "Indefinido":
            p += 0.05

        # Modalidad (Remoto/Híbrido son más atractivos actualmente)
        if row["modalidad"] in ["Remoto", "Hibrido"]:
            p += 0.05

        prob[i] = np.clip(p + np.random.normal(0, 0.05), 0, 1)

    df["aceptado"] = (prob > 0.55).astype(int)

    print(f"✓ Dataset generado: {n_samples} muestras")
    print(f"  Aceptados:     {df['aceptado'].sum()} ({df['aceptado'].mean()*100:.1f}%)")
    print(f"  No aceptados:  {(df['aceptado'] == 0).sum()} ({(1 - df['aceptado'].mean())*100:.1f}%)")
    return df


# ─── Preprocesamiento ───────────────────────────────────────────────────────────

class WorkwisePreprocessor:
    """Encoders y scaler para las variables del modelo."""

    def __init__(self):
        self.encoders: dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.categorical_cols = ["tipo_empleo", "modalidad", "tipo_contrato", "nivel_estudio", "sector"]
        self.numeric_cols = ["experiencia"]

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        features = self._encode_categoricals(df, fit=True)
        features = self._scale_numerics(features, df, fit=True)
        return features

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        features = self._encode_categoricals(df, fit=False)
        features = self._scale_numerics(features, df, fit=False)
        return features

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        result = df.copy()
        for col in self.categorical_cols:
            if fit:
                le = LabelEncoder()
                result[col] = le.fit_transform(df[col])
                self.encoders[col] = le
            else:
                le = self.encoders[col]
                result[col] = le.transform(df[col])
        return result

    def _scale_numerics(self, df: pd.DataFrame, original: pd.DataFrame, fit: bool) -> np.ndarray:
        numeric_data = original[self.numeric_cols].values
        if fit:
            scaled = self.scaler.fit_transform(numeric_data)
        else:
            scaled = self.scaler.transform(numeric_data)

        cat_data = df[self.categorical_cols].values
        return np.hstack([cat_data, scaled])

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.encoders, f"{directory}/encoders.pkl")
        joblib.dump(self.scaler,   f"{directory}/scaler.pkl")
        print(f"✓ Preprocesador guardado en '{directory}/'")

    def load(self, directory: str):
        self.encoders = joblib.load(f"{directory}/encoders.pkl")
        self.scaler   = joblib.load(f"{directory}/scaler.pkl")
        print(f"✓ Preprocesador cargado desde '{directory}/'")


# ─── Red Neuronal ───────────────────────────────────────────────────────────────

def build_model(input_dim: int) -> keras.Model:
    """
    Red neuronal para clasificación binaria.
    Arquitectura: 6 entradas → 128 → 64 → 32 → 1 salida (sigmoide)
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(32, activation="relu"),
        layers.Dropout(0.1),

        layers.Dense(1, activation="sigmoid")
    ], name="workwise_predictor")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    return model


# ─── Entrenamiento ──────────────────────────────────────────────────────────────

def train():
    print("\n══════════════════════════════════════════")
    print("   WORKWISE — Entrenamiento del Modelo")
    print("══════════════════════════════════════════\n")

    # 1. Generar datos
    df = generate_synthetic_data(n_samples=8000)

    # 2. Preprocesar
    preprocessor = WorkwisePreprocessor()
    X = preprocessor.fit_transform(df.drop(columns=["aceptado"]))
    y = df["aceptado"].values

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n✓ Train: {len(X_train)} | Test: {len(X_test)}")

    # 4. Construir modelo
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
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=1
    )

    # 7. Evaluar
    print("\n── Evaluación en Test ──────────────────────")
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=["No Aceptado", "Aceptado"]))
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    # 8. Guardar
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(f"{MODEL_DIR}/model.keras")
    preprocessor.save(MODEL_DIR)

    # Guardar metadatos
    metadata = {
        "input_features": ["tipo_empleo", "modalidad", "tipo_contrato",
                            "nivel_estudio", "sector", "experiencia"],
        "output": "aceptado (0 = No, 1 = Sí)",
        "threshold": 0.5,
        "accuracy_test": round(float(accuracy_score(y_test, y_pred)), 4),
        "categoricals": {
            "tipo_empleo":   TIPO_EMPLEO,
            "modalidad":     MODALIDAD,
            "tipo_contrato": TIPO_CONTRATO,
            "nivel_estudio": NIVEL_ESTUDIO,
            "sector":        SECTOR
        }
    }
    with open(f"{MODEL_DIR}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Modelo guardado en '{MODEL_DIR}/'")
    print("══════════════════════════════════════════\n")


if __name__ == "__main__":
    train()

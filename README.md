# Workwise ML — Red Neuronal de Predicción Laboral

Predice si un postulante será **aceptado o no** en una oferta laboral,
basándose en sus características de perfil.

---

## Estructura del proyecto

```
workwise_ml/
├── model.py          ← Generación de datos, entrenamiento y guardado del modelo
├── api.py            ← API REST con FastAPI (consume Java y Flutter)
├── requirements.txt  ← Dependencias
└── saved_model/      ← Generado al entrenar
    ├── model.keras
    ├── encoders.pkl
    ├── scaler.pkl
    └── metadata.json
```

---

## Instalación

```bash
pip install -r requirements.txt
```

---

## Uso

### 1. Entrenar el modelo

```bash
python model.py
```

Genera datos sintéticos, entrena la red neuronal y guarda el modelo en `saved_model/`.

### 2. Levantar la API

```bash
python api.py
# o con uvicorn directamente:
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Documentación interactiva disponible en: **http://localhost:8000/docs**

---

## Endpoints

| Método | Ruta             | Descripción                        |
|--------|------------------|------------------------------------|
| GET    | `/health`        | Estado del servicio                |
| GET    | `/metadata`      | Variables y valores aceptados      |
| POST   | `/predict`       | Predice un postulante              |
| POST   | `/predict/batch` | Predice una lista de postulantes   |

---

## Ejemplo de request

**POST /predict**
```json
{
  "tipo_empleo":   "Tiempo_Completo",
  "modalidad":     "Remoto",
  "tipo_contrato": "Indefinido",
  "experiencia":   4,
  "nivel_estudio": "Universitario",
  "sector":        "Tecnologia"
}
```

**Response**
```json
{
  "aceptado":     true,
  "probabilidad": 0.8312,
  "confianza":    "Alta",
  "mensaje":      "El postulante tiene alta probabilidad de ser aceptado."
}
```

---

## Integración con Spring Boot (Java)

```java
// Dependencia: spring-boot-starter-web

@Service
public class WorkwiseMLService {

    private final RestTemplate restTemplate = new RestTemplate();
    private final String ML_URL = "http://localhost:8000/predict";

    public PrediccionResponse predecir(PostulanteDTO postulante) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        Map<String, Object> body = Map.of(
            "tipo_empleo",   postulante.getTipoEmpleo(),
            "modalidad",     postulante.getModalidad(),
            "tipo_contrato", postulante.getTipoContrato(),
            "experiencia",   postulante.getExperiencia(),
            "nivel_estudio", postulante.getNivelEstudio(),
            "sector",        postulante.getSector()
        );

        HttpEntity<Map<String, Object>> request = new HttpEntity<>(body, headers);
        return restTemplate.postForObject(ML_URL, request, PrediccionResponse.class);
    }
}
```

---

## Integración con Flutter (Dart/Dio)

```dart
// pubspec.yaml: dio: ^5.4.0

class WorkwiseMLService {
  final Dio _dio = Dio(BaseOptions(baseUrl: 'http://TU_IP:8000'));

  Future<PrediccionResponse> predecir(Postulante postulante) async {
    final response = await _dio.post('/predict', data: postulante.toJson());
    return PrediccionResponse.fromJson(response.data);
  }
}

class PrediccionResponse {
  final bool aceptado;
  final double probabilidad;
  final String confianza;
  final String mensaje;

  PrediccionResponse.fromJson(Map<String, dynamic> json)
      : aceptado     = json['aceptado'],
        probabilidad = json['probabilidad'],
        confianza    = json['confianza'],
        mensaje      = json['mensaje'];
}
```

---

## Variables del modelo

| Variable        | Tipo    | Valores aceptados                                                                 |
|-----------------|---------|-----------------------------------------------------------------------------------|
| `tipo_empleo`   | String  | Tiempo_Completo, Medio_Tiempo, Por_Horas, Freelance                               |
| `modalidad`     | String  | Presencial, Remoto, Hibrido                                                       |
| `tipo_contrato` | String  | Indefinido, Practicas, Obra_Labor, Fijo                                           |
| `experiencia`   | Integer | 0 – 50 (años)                                                                     |
| `nivel_estudio` | String  | Sin_estudios, Bachiller, Tecnico_Tecnologo, Tecnologo_Universitario, Universitario, Master, Doctorado |
| `sector`        | String  | Aeroespacial, Agricultura, Agroindustria, Ambiental, ArtesCreativas, Automotriz, BienesRaices, Biotecnologia, Comercio, Construccion, Consultoria, Deportes, Diseno, Educacion, Energia, Finanzas, Gobierno, Investigacion, Legal, Logistica, Manufactura, Marketing, Medios, Mineria, Naval, Quimica, RRHH, Salud, Seguros, Servicios, Social, Tecnologia, Textil, Transporte, Turismo |

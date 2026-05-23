from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import io

# -----------------------------------
# Inicializar aplicación
# -----------------------------------
app = FastAPI(title="API Predicción MNIST")

# -----------------------------------
# Configuración CORS
# -----------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción usa tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------
# Cargar modelo UNA sola vez
# -----------------------------------
try:
    modelo = tf.keras.models.load_model("api/model/modeloV3.keras")
    print("Modelo cargado correctamente")
except Exception as e:
    raise RuntimeError(f"Error cargando el modelo: {e}")

# -----------------------------------
# Clases
# -----------------------------------
nombre_clases = [str(i) for i in range(10)]

# -----------------------------------
# Esquema opcional de entrada
# -----------------------------------
class ImagenInput(BaseModel):
    imagen: list

# -----------------------------------
# Ruta raíz
# -----------------------------------
@app.get("/")
def root():
    return {
        "message": "Modelo de IA listo y funcionando",
        "version": "1.0",
        "autor": "Leonardo Martínez González"
    }

# -----------------------------------
# Ruta de predicción
# -----------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Leer imagen
        contents = await file.read()

        # Abrir imagen en escala de grises
        image = Image.open(io.BytesIO(contents)).convert("L")

        # Redimensionar a 28x28
        image = image.resize((28, 28))

        # Convertir a arreglo numpy
        image = np.array(image).astype("float32") / 255.0

        # Invertir colores
        image = 1.0 - image

        # Agregar dimensiones necesarias
        image = np.expand_dims(image, axis=-1)  # (28,28,1)
        image = np.expand_dims(image, axis=0)   # (1,28,28,1)

        # Predicción
        pred = modelo.predict(image)

        # Obtener clase y probabilidad
        clase_idx = int(np.argmax(pred))
        probabilidad = float(np.max(pred))

        print(f"Clase predicha: {clase_idx}")
        print(f"Probabilidad: {probabilidad}")

        return {
            "clase": str(clase_idx),
            "probabilidad": probabilidad
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

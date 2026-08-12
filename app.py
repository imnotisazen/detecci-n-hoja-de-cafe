import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from groq import Groq
import os


# ==========================================
# CONFIGURACIÓN
# ==========================================

st.set_page_config(
    page_title="Detector de Enfermedades del Café",
    page_icon="☕",
    layout="centered"
)


# ==========================================
# TÍTULO
# ==========================================

st.title("☕ Detector de Enfermedades en Hojas de Café")

st.write(
    "Carga una imagen de una hoja de café y el sistema "
    "utilizará inteligencia artificial para identificar "
    "posibles enfermedades y generar recomendaciones técnicas."
)


# ==========================================
# CARGAR MODELO
# ==========================================

@st.cache_resource
def cargar_modelo():

    return tf.keras.models.load_model(
    "modelo_cafe.h5"
)


modelo = cargar_modelo()


# ==========================================
# CLASES
# ==========================================

clases = [
    "Healthy",
    "Leaf_Miner",
    "Phoma",
    "Rust"
]


nombres_espanol = {
    "Healthy": "Hoja saludable",
    "Leaf_Miner": "Minador de la hoja",
    "Phoma": "Phoma",
    "Rust": "Roya"
}


# ==========================================
# API GROQ
# ==========================================

api_key = os.getenv("GROQ_API_KEY")

if api_key:

    cliente_groq = Groq(
        api_key=api_key
    )

else:

    cliente_groq = None


# ==========================================
# SUBIR IMAGEN
# ==========================================

archivo = st.file_uploader(
    "📷 Selecciona una imagen de una hoja de café",
    type=["jpg", "jpeg", "png"]
)


# ==========================================
# PROCESAR IMAGEN
# ==========================================

if archivo is not None:

    imagen = Image.open(archivo).convert("RGB")

    st.image(
        imagen,
        caption="Imagen seleccionada",
        use_container_width=True
    )


    if st.button(
        "🔍 Analizar hoja",
        use_container_width=True
    ):

        with st.spinner(
            "Analizando la hoja..."
        ):

            # Preparar imagen
            imagen_modelo = imagen.resize(
                (224, 224)
            )

            imagen_array = np.array(
                imagen_modelo
            ) / 255.0

            imagen_array = np.expand_dims(
                imagen_array,
                axis=0
            )


            # Predicción
            prediccion = modelo.predict(
                imagen_array,
                verbose=0
            )

            indice = np.argmax(
                prediccion[0]
            )

            clase = clases[indice]

            confianza = (
                prediccion[0][indice] * 100
            )


        # ======================================
        # RESULTADO
        # ======================================

        st.subheader(
            "🔍 Resultado del análisis"
        )

        st.success(
            f"**{nombres_espanol[clase]}**"
        )

        st.metric(
            "Confianza de la predicción",
            f"{confianza:.2f}%"
        )


        # ======================================
        # GROQ
        # ======================================

        if cliente_groq is not None:

            with st.spinner(
                "Generando orientación técnica..."
            ):

                prompt = f"""
Eres un especialista técnico en el cultivo de café.

Un modelo de inteligencia artificial analizó
una imagen de una hoja de café.

Resultado de la inteligencia artificial:

Enfermedad detectada:
{nombres_espanol[clase]}

Confianza:
{confianza:.2f}%

Genera una orientación técnica clara y organizada
para el productor de café.

Incluye exactamente estas secciones:

1. Descripción de la enfermedad
2. Síntomas principales
3. Manejo preventivo
4. Buenas prácticas para el cultivo
5. Seguimiento y monitoreo

La información debe ser práctica, clara y fácil
de comprender.

Aclara que la predicción corresponde a una
herramienta de inteligencia artificial y que,
cuando sea necesario, debe confirmarse mediante
observación de campo o con un especialista.
"""


                respuesta = cliente_groq.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Eres un asistente técnico "
                                "especializado en cultivo de café."
                            )
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3
                )


                recomendaciones = (
                    respuesta
                    .choices[0]
                    .message
                    .content
                )


            st.subheader(
                "🧠 Orientación técnica"
            )

            st.markdown(
                recomendaciones
            )


        else:

            st.warning(
                "No se encontró la API Key de Groq."
            )

            st.info(
                "Configura GROQ_API_KEY en los Secrets "
                "de Streamlit Cloud."
            )

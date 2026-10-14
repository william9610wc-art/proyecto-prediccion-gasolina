from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Predicción de gasolina", page_icon="⛽", layout="centered")
st.title("Predicción del precio de la gasolina")
st.write(
    "Aplicación de regresión lineal múltiple. Ingresa las variables económicas, "
    "energéticas y geopolíticas para estimar el precio en USD por galón."
)
st.warning(
    "La estimación depende de la calidad del dataset suministrado; no es una cotización "
    "oficial ni garantiza el precio futuro del mercado."
)

CARPETA = Path(__file__).resolve().parent
NOMBRE_CSV = "dataset_precio_gasolina (1) (3).csv"
RUTA_CSV = CARPETA / NOMBRE_CSV
RUTA_MODELO = CARPETA / "modelo_gasolina.pkl"

if not RUTA_CSV.exists():
    st.error(f"No se encontró el archivo de datos: {RUTA_CSV.name}")
    st.info("Guarda el CSV en la misma carpeta que app_gasolina.py.")
    st.stop()

if not RUTA_MODELO.exists():
    st.error("No se encontró modelo_gasolina.pkl.")
    st.info(
        "Abre Proyecto_Regresion_Gasolina.ipynb, ejecuta todas las celdas y vuelve a "
        "iniciar esta aplicación. El notebook crea el archivo del modelo."
    )
    st.stop()

try:
    df = pd.read_csv(RUTA_CSV)
    if "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    for columna in df.columns:
        if columna != "Fecha":
            df[columna] = pd.to_numeric(df[columna], errors="coerce")
    df = df.drop_duplicates()
    df = df.dropna(subset=["Precio_Gasolina_USD_Galon"])
    df = df[df["Precio_Gasolina_USD_Galon"] > 0]
    paquete = joblib.load(RUTA_MODELO)
    modelo = paquete["modelo"]
    caracteristicas = paquete["caracteristicas"]
    limites_atipicos = paquete.get("limites_atipicos", {})
    margen_95 = float(paquete.get("margen_intervalo_95", 0.0))
    metricas = paquete.get("metricas", {})
except Exception as error:
    st.error(f"No se pudieron cargar los datos o el modelo: {error}")
    st.info(
        "Comprueba que ejecutaste el notebook y que las librerías están instaladas "
        "con requirements_gasolina.txt. Si cambiaste la versión de scikit-learn, vuelve "
        "a ejecutar el notebook para generar modelo_gasolina.pkl en tu equipo."
    )
    st.stop()

faltantes = [columna for columna in caracteristicas if columna not in df.columns]
if faltantes:
    st.error(f"El CSV no contiene estas variables del modelo: {faltantes}")
    st.stop()

st.subheader("1. Ingresa los datos")
st.caption(
    "Los campos empiezan con la mediana observada en el CSV. Puedes cambiarlos para "
    "simular un escenario. Las variables continuas se limitan a los rangos IQR aprendidos "
    "en el entrenamiento para reducir el efecto de valores extremos."
)
valores = {}
with st.form("formulario_prediccion"):
    for columna in caracteristicas:
        serie = df[columna].dropna()
        predeterminado = float(serie.median()) if not serie.empty else 0.0
        # Evitar NaN como valor inicial en el control numérico.
        if not np.isfinite(predeterminado):
            predeterminado = 0.0
        valores[columna] = st.number_input(
            columna.replace("_", " "),
            value=predeterminado,
            format="%.3f",
            key=f"entrada_{columna}",
        )
    enviar = st.form_submit_button("Predecir precio")

if enviar:
    entrada = pd.DataFrame([valores], columns=caracteristicas)
    for columna, limites in limites_atipicos.items():
        if columna in entrada.columns:
            entrada[columna] = entrada[columna].clip(lower=limites[0], upper=limites[1])
    try:
        prediccion = float(modelo.predict(entrada)[0])
        st.subheader("2. Resultado")
        st.metric("Precio estimado", f"${prediccion:.2f} USD/galón")
        st.write(
            "**Intervalo empírico aproximado del 95%:** "
            f"${prediccion - margen_95:.2f} a ${prediccion + margen_95:.2f} USD/galón"
        )
        st.caption(
            "El intervalo se calibra con los errores absolutos del conjunto de prueba; "
            "es una estimación empírica y no un intervalo exacto para la media."
        )
    except Exception as error:
        st.error(f"No fue posible generar la predicción: {error}")

st.divider()
st.subheader("Métricas del modelo en el conjunto de prueba")
col1, col2 = st.columns(2)
if metricas:
    col1.metric("R²", f"{metricas.get('R2', float('nan')):.4f}")
    col2.metric("MAE (USD/galón)", f"{metricas.get('MAE', float('nan')):.4f}")
    col3, col4 = st.columns(2)
    col3.metric("MSE", f"{metricas.get('MSE', float('nan')):.4f}")
    col4.metric("RMSE (USD/galón)", f"{metricas.get('RMSE', float('nan')):.4f}")
else:
    st.info("Las métricas no están disponibles en el archivo del modelo.")

with st.expander("Ver datos cargados"):
    st.dataframe(df, use_container_width=True)

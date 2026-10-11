import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_prep import enriquecer_datos_temporales
from predict import entrenar_random_forest, calcular_metricas
from optimize import optimizar_abastecimiento

st.set_page_config(page_title="Optimizador Logístico", layout="wide", page_icon="📦")

st.title("📦 Plataforma de Inteligencia Logística")
st.markdown("Integración de Machine Learning (Random Forest) y MILP (Gurobi) adaptado al marco metodológico PRISMA.")
st.divider()

# --- BARRA LATERAL: INPUTS DEL USUARIO ---
st.sidebar.header("⚙️ Parámetros Financieros (COP)")
c_orden = st.sidebar.number_input("Costo por Emitir Orden ($)", value=50000, step=1000)
c_compra = st.sidebar.number_input("Costo de Compra por Caja ($)", value=45000, step=1000)
c_inv = st.sidebar.number_input("Costo de Almacenamiento ($)", value=500, step=100)
c_faltante = st.sidebar.number_input("Penalización por Faltante ($)", value=80000, step=1000)
capacidad = st.sidebar.number_input("Capacidad de Bodega (m3)", value=500, step=50)

st.sidebar.divider()
st.sidebar.header("📅 Horizonte de Planificación")
# Nuevo control deslizante para que el usuario elija los días a predecir
dias_proyeccion = st.sidebar.slider("Días a Proyectar", min_value=1, max_value=30, value=7)

# --- PANEL PRINCIPAL ---
st.subheader("1. Carga de Datos y Selección de Producto")
archivo_subido = st.file_uploader("Arrastra tu archivo Excel con el historial de ventas", type=["xlsx", "csv"])

if archivo_subido is not None:
    if archivo_subido.name.endswith('.csv'):
        df = pd.read_csv(archivo_subido)
    else:
        df = pd.read_excel(archivo_subido)
        
    lista_productos = df['sku'].unique().tolist()
    producto_seleccionado = st.selectbox("Seleccione el producto a optimizar:", lista_productos)
    
    st.divider()
    
    if st.button(f"🚀 Ejecutar IA y Optimización para: {producto_seleccionado}", use_container_width=True):
        
        with st.spinner("Calculando variables rezagadas (Lags) y calendario..."):
            df_procesado = enriquecer_datos_temporales(df, 'fecha')
            
        with st.spinner("Entrenando modelo Random Forest Regressor..."):
            modelo_rf, X_test, y_test, predicciones = entrenar_random_forest(df_procesado, sku_objetivo=producto_seleccionado)
            mae, rmse, mape, wape = calcular_metricas(y_test, predicciones)
            
            st.subheader("📊 Rendimiento Predictivo (Ecuaciones 3-6)")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAE (Cajas)", f"{mae:.2f}")
            col2.metric("RMSE (Cajas)", f"{rmse:.2f}")
            col3.metric("MAPE (%)", f"{mape:.2f}%")
            col4.metric("WAPE (%)", f"{wape:.2f}%")
            
        with st.spinner(f"Ejecutando solver Gurobi para {dias_proyeccion} días..."):
            # Aquí aplicamos la cantidad de días que el usuario eligió en el slider
            horizonte_prediccion = predicciones[-dias_proyeccion:] 
            
            tabla_resultados, costo_total = optimizar_abastecimiento(
                horizonte_prediccion, c_orden, c_compra, c_inv, c_faltante, capacidad
            )
            
            st.divider()
            # El título de la tabla se actualiza dinámicamente
            st.subheader(f"📋 Plan Prescriptivo de Abastecimiento (Próximos {dias_proyeccion} días)")
            st.metric(label="Costo Total Logístico Proyectado (Función Objetivo)", value=f"${costo_total:,.0f} COP")
            
            st.dataframe(tabla_resultados, use_container_width=True, hide_index=True)
            st.success("¡Pipeline matemático ejecutado con éxito!")
else:
    st.info("Esperando la carga del archivo de ventas histórico...")
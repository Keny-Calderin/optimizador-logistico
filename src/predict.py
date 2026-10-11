import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import os
import sys

# Conectar con nuestro nuevo data_prep
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_prep import cargar_datos_historicos, enriquecer_datos_temporales

def calcular_metricas(y_true, y_pred):
    """Calcula las 4 métricas exigidas en el Marco Teórico (Eq 3-6)."""
    # Eq 3: MAE (Error Absoluto Medio)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Eq 4: RMSE (Raíz del Error Cuadrático Medio)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Eq 5: MAPE (Error Porcentual Absoluto Medio)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    # Eq 6: WAPE (Error Porcentual Absoluto Ponderado)
    wape = (np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)) * 100
    
    return mae, rmse, mape, wape

def entrenar_random_forest(df, sku_objetivo='Águila'):
    print(f"Preparando matriz de entrenamiento (Random Forest) para: {sku_objetivo}...")
    df_sku = df[df['sku'] == sku_objetivo].copy()
    
    # Seleccionamos las variables (X) alineadas con la Eq 2: Lags + Calendario
    caracteristicas = [
        'año', 'mes', 'dia_semana', 'es_quincena', 'es_fin_de_semana', 
        'precio_caja', 'ventas_lag_1', 'ventas_lag_7', 'media_movil_7d'
    ]
    
    X = df_sku[caracteristicas]
    y = df_sku['cantidad_vendida']
    
    # MUY IMPORTANTE: Para pronóstico de demanda (series de tiempo) no se cruzan 
    # los datos al azar. Se toma el pasado para predecir el futuro (shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print("Entrenando el modelo Random Forest Regressor (Ensamble tipo Bagging)...")
    modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo_rf.fit(X_train, y_train)
    
    print("Evaluando predicciones...")
    predicciones = modelo_rf.predict(X_test)
    
    # Calculamos las métricas exactas
    mae, rmse, mape, wape = calcular_metricas(y_test, predicciones)
    
    print("\n" + "="*50)
    print("MÉTRICAS DE EVALUACIÓN PREDICTIVA (Eq 3 - 6)")
    print("="*50)
    print(f"1. Error Absoluto Medio (MAE): {mae:.2f} cajas")
    print(f"2. Raíz Error Cuadrático Medio (RMSE): {rmse:.2f} cajas")
    print(f"3. Error Porcentual Absoluto Medio (MAPE): {mape:.2f}%")
    print(f"4. Error Porcentual Absoluto Ponderado (WAPE): {wape:.2f}%")
    print("="*50)
    
    return modelo_rf, X_test, y_test, predicciones

if __name__ == "__main__":
    ruta_raw = "data/raw/historico_ventas_simulado.xlsx"
    
    if os.path.exists(ruta_raw):
        # 1. Pipeline de datos
        datos = cargar_datos_historicos(ruta_raw)
        datos_enriquecidos = enriquecer_datos_temporales(datos, 'fecha')
        
        # 2. Motor de Machine Learning Clásico
        sku_piloto = 'Águila'
        modelo, X_test, y_test, predicciones = entrenar_random_forest(datos_enriquecidos, sku_objetivo=sku_piloto)
        
        # 3. Prueba de fuego
        print(f"\n--- PRUEBA DE PREDICCIÓN PARA LOS ÚLTIMOS 5 DÍAS DE {sku_piloto.upper()} ---")
        ultimos_y = y_test.tail(5).values
        ultimos_pred = predicciones[-5:]
        
        for i in range(5):
            real = ultimos_y[i]
            pred = int(ultimos_pred[i])
            print(f"Venta Real: {real} cajas  |  Predicción RF: {pred} cajas")
    else:
        print(f"No se encontró el archivo en {ruta_raw}.")
import pandas as pd
import numpy as np

def cargar_datos_historicos(ruta_archivo):
    print(f"Cargando datos desde {ruta_archivo}...")
    df = pd.read_excel(ruta_archivo)
    return df

def enriquecer_datos_temporales(df, columna_fecha='fecha'):
    print("Enriqueciendo el dataset con variables de calendario (Z) y rezagos históricos (X)...")
    
    # 1. Asegurar formato de fecha y ordenar cronológicamente por producto
    df[columna_fecha] = pd.to_datetime(df[columna_fecha])
    df = df.sort_values(by=['sku', columna_fecha]).reset_index(drop=True)
    
    # 2. Variables contextuales y de calendario (Z_{t+k})
    df['año'] = df[columna_fecha].dt.year
    df['mes'] = df[columna_fecha].dt.month
    df['dia_semana'] = df[columna_fecha].dt.dayofweek
    df['es_fin_de_semana'] = df['dia_semana'].apply(lambda x: 1 if x >= 4 else 0)
    df['es_quincena'] = df[columna_fecha].dt.day.apply(lambda x: 1 if x in [14, 15, 16, 29, 30, 31] else 0)
    
    # 3. Variables históricas rezagadas (X_{i, t-p})
    # 'shift' empuja los datos hacia abajo para alinear la venta de ayer con la fila de hoy
    df['ventas_lag_1'] = df.groupby('sku')['cantidad_vendida'].shift(1) # t-1 (Ayer)
    df['ventas_lag_7'] = df.groupby('sku')['cantidad_vendida'].shift(7) # t-7 (Hace una semana)
    
    # Promedio móvil de los últimos 7 días como indicador de tendencia
    df['media_movil_7d'] = df.groupby('sku')['cantidad_vendida'].transform(
        lambda x: x.shift(1).rolling(window=7).mean()
    )
    
    # 4. Limpieza de datos nulos
    # Al desplazar 7 días, las primeras 7 filas de cada producto quedarán vacías (NaN)
    df = df.dropna().reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    ruta_raw = "data/raw/historico_ventas_simulado.xlsx"
    
    datos = cargar_datos_historicos(ruta_raw)
    datos_procesados = enriquecer_datos_temporales(datos)
    
    print("\nVista previa del dataset alineado con la Ecuación 2:")
    print(datos_procesados[['fecha', 'sku', 'cantidad_vendida', 'ventas_lag_1', 'ventas_lag_7', 'media_movil_7d']].head(10))
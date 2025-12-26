"""
Configuración global del proyecto
Definición de rutas, constantes y configuración inicial
"""

import os
import json
import pickle
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIGURACIÓN DE RUTAS
# =============================================================================

# Directorio base del proyecto
BASE_DIR = Path(__file__).parent.parent

# Directorios principales
DIRS = {
    'datos': BASE_DIR / 'data' / 'processed',
    'raw': BASE_DIR / 'data' / 'raw',
    'resultados': BASE_DIR / 'data' / 'results',
    'modelos': BASE_DIR / 'models',
    'visualizaciones': BASE_DIR / 'reports' / 'figures',
    'metricas': BASE_DIR / 'reports' / 'metrics'
}

# Crear directorios si no existen
for dir_path in DIRS.values():
    dir_path.mkdir(parents=True, exist_ok=True)

# Archivo de datos
DATASET_PATH = "data/raw/dataset_FINAL_IMPUTADO_20251201_1356.xlsx"


# =============================================================================
# CONSTANTES
# =============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.30
N_FOLDS = 10

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def guardar_resultados_json(datos, nombre_archivo):
    """Guarda resultados en formato JSON"""
    ruta = DIRS['metricas'] / f'{nombre_archivo}.json'
    with open(ruta, 'w') as f:
        json.dump(datos, f, indent=4)
    print(f"   Guardado: {nombre_archivo}.json")

def cargar_modelo(nombre_modelo):
    """Carga un modelo guardado"""
    ruta = DIRS['modelos'] / f'{nombre_modelo}.pkl'
    with open(ruta, 'rb') as f:
        return pickle.load(f)

def guardar_modelo(modelo, nombre_modelo):
    """Guarda un modelo"""
    ruta = DIRS['modelos'] / f'{nombre_modelo}.pkl'
    with open(ruta, 'wb') as f:
        pickle.dump(modelo, f)
    print(f"   Modelo guardado: {nombre_modelo}.pkl")

# Mensaje de confirmación
print("Configuración cargada correctamente")
print(f"Directorio base: {BASE_DIR}")
print(f"Dataset: {DATASET_PATH}")
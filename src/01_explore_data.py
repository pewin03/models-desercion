"""
BLOQUE: CARGA Y CONFIGURACIÓN INICIAL
Carga del dataset y configuración del entorno
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import sys

warnings.filterwarnings('ignore')

# Importar configuración (CORREGIDO)
sys.path.insert(0, str(Path(__file__).parent))
exec(open(Path(__file__).parent / '00_config.py').read())

print("="*80)
print("BLOQUE: CARGA Y CONFIGURACIÓN INICIAL")
print("="*80)

# -----------------------------------------------------------------------------
#Configuración de visualizaciones
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("PASO 1: CONFIGURACIÓN DE VISUALIZACIONES")
print("-"*80)

plt.style.use('default')
sns.set_palette("husl")

# Configuración de matplotlib
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

print("Configuración de visualizaciones aplicada")
print(f"   Estilo: default")
print(f"   Paleta: husl")
print(f"   Tamaño figura: {plt.rcParams['figure.figsize']}")

# -----------------------------------------------------------------------------
#Carga del dataset
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("CARGA DEL DATASET")
print("-"*80)

print(f"\nCargando dataset desde:")
print(f"   {DATASET_PATH}")

try:
    df = pd.read_excel(DATASET_PATH)
    print(f"\nDataset cargado exitosamente")
    print(f"   Registros: {len(df):,}")
    print(f"   Columnas: {len(df.columns)}")
    
except FileNotFoundError:
    print(f"\nERROR: Archivo no encontrado")
    print(f"Verifica que el archivo existe en: {DATASET_PATH}")
    sys.exit(1)
except Exception as e:
    print(f"\nERROR al cargar el archivo: {str(e)}")
    sys.exit(1)

# -----------------------------------------------------------------------------
#Normalización de nombres de columnas
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("NORMALIZACIÓN DE NOMBRES DE COLUMNAS")
print("-"*80)

print("\nNormalizando nombres de columnas...")

# Función de normalización
def normalizar_columna(col):
    col = str(col).strip()
    col = col.replace(' ', '_')
    col = col.replace('-', '_')
    col = col.replace('(', '').replace(')', '')
    col = col.replace('/', '_')
    col = col.upper()
    return col

# Aplicar normalización
df.columns = [normalizar_columna(col) for col in df.columns]

print(f"   Columnas normalizadas: {len(df.columns)}")

# Mostrar primeras 15 columnas
print(f"\nPrimeras 15 columnas:")
for i, col in enumerate(df.columns[:15], 1):
    print(f"   {i:2d}. {col}")

# Mostrar TODAS las columnas (para tu análisis)
print(f"\nTODAS LAS COLUMNAS ({len(df.columns)}):")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col}")

# -----------------------------------------------------------------------------
# Paso 4: Información básica del dataset
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("INFORMACIÓN BÁSICA DEL DATASET")
print("-"*80)

print(f"\nDimensiones del dataset:")
print(f"   Filas (registros): {df.shape[0]:,}")
print(f"   Columnas (variables): {df.shape[1]}")

print(f"\nMemoria utilizada:")
memoria_mb = df.memory_usage(deep=True).sum() / (1024**2)
print(f"   {memoria_mb:.2f} MB")

print(f"\nTipos de datos:")
tipos_datos = df.dtypes.value_counts()
for tipo, cantidad in tipos_datos.items():
    print(f"   {tipo}: {cantidad} columnas")

# -----------------------------------------------------------------------------
#Vista previa de los datos
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("VISTA PREVIA DE LOS DATOS")
print("-"*80)

print("\nPrimeras 3 filas del dataset:")
print(df.head(3))

print("\nÚltimas 3 filas del dataset:")
print(df.tail(3))

# -----------------------------------------------------------------------------
#Verificar columnas clave
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("VERIFICACIÓN DE COLUMNAS CLAVE")
print("-"*80)

columnas_esperadas = ['ESTUDIANTE', 'ESTADO_FINAL', 'PERIODO', 'AÑO', 'SEMESTRE']
print("\nBuscando columnas clave...")

for col_esperada in columnas_esperadas:
    # Buscar variaciones
    encontrada = None
    for col in df.columns:
        if col_esperada in col:
            encontrada = col
            break
    
    if encontrada:
        print(f"   ✓ {col_esperada}: Encontrada como '{encontrada}'")
    else:
        print(f"   ✗ {col_esperada}: NO encontrada")

# -----------------------------------------------------------------------------
#Guardar dataset normalizado
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("GUARDANDO DATASET NORMALIZADO")
print("-"*80)

ruta_guardado = DIRS['datos'] / 'dataset_normalizado.csv'
df.to_csv(ruta_guardado, index=False)

print(f"\nDataset normalizado guardado en:")
print(f"   {ruta_guardado}")

# -----------------------------------------------------------------------------
# Resumen
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("RESUMEN - BLOQUE COMPLETADO")
print("="*80)

print(f"""
DATASET CARGADO Y NORMALIZADO

Archivo fuente: {Path(DATASET_PATH).name}
Registros: {len(df):,}
Columnas: {len(df.columns)}
Memoria: {memoria_mb:.2f} MB

Dataset normalizado guardado en:
   {ruta_guardado}

PRÓXIMO PASO:
   Ejecutar src/02_analisis_desercion.py (BLOQUE 2)
""")

print("="*80)
print("BLOQUE COMPLETADO EXITOSAMENTE")
print("="*80)

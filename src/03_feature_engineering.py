"""
FEATURE ENGINEERING Y PREPARACIÓN PARA MODELADO
Optimizado para predicción de deserción e identificación de factores clave
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
import pickle

warnings.filterwarnings('ignore')

# Importar configuración
sys.path.insert(0, str(Path(__file__).parent))
exec(open(Path(__file__).parent / '00_config.py').read())

print("="*80)
print("FEATURE ENGINEERING Y PREPARACIÓN PARA MODELADO")
print("Objetivo: Predecir deserción e identificar factores determinantes")
print("="*80)

# -----------------------------------------------------------------------------
#Cargar dataset base
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("VERIFICACIÓN DE DATASET BASE")
print("-"*80)

ruta_dataset = DIRS['datos'] / 'dataset_base_modelado_corregido.csv'
print(f"\nCargando dataset base desde: {ruta_dataset}")

df_base = pd.read_csv(ruta_dataset)

print(f"\nDataset base cargado:")
print(f"   Registros: {len(df_base):,}")
print(f"   Columnas: {len(df_base.columns)}")

if 'DESERCION' not in df_base.columns:
    raise ValueError("ERROR: Variable DESERCION no encontrada")

desertores = df_base['DESERCION'].sum()
no_desertores = len(df_base) - desertores

print(f"\nDistribución de variable objetivo:")
print(f"   No desertores: {no_desertores:,} ({no_desertores/len(df_base)*100:.2f}%)")
print(f"   Desertores:    {desertores:,} ({desertores/len(df_base)*100:.2f}%)")
print(f"   Ratio:         {no_desertores/desertores:.2f}:1")

# -----------------------------------------------------------------------------
#Cargar dataset completo para períodos cursados
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("ANÁLISIS DE PERÍODOS CURSADOS")
print("-"*80)

ruta_completo = DIRS['datos'] / 'dataset_con_desercion_corregido.csv'
df_completo = pd.read_csv(ruta_completo)

print("\nCalculando períodos cursados por estudiante...")

df_periodos_cursados = df_completo.groupby('ESTUDIANTE').size().reset_index(name='NUM_PERIODOS_CURSADOS')

# Merge con df_base
df_base['ESTUDIANTE'] = df_base['ESTUDIANTE'].astype(int)
df_periodos_cursados['ESTUDIANTE'] = df_periodos_cursados['ESTUDIANTE'].astype(int)

if 'NUM_PERIODOS_CURSADOS' in df_base.columns:
    df_base = df_base.drop(columns=['NUM_PERIODOS_CURSADOS'])

df_base = df_base.merge(df_periodos_cursados, on='ESTUDIANTE', how='left')

print(f"   Variable NUM_PERIODOS_CURSADOS creada")

print(f"\nEstadísticas:")
print(f"   Media:    {df_base['NUM_PERIODOS_CURSADOS'].mean():.2f}")
print(f"   Mediana:  {df_base['NUM_PERIODOS_CURSADOS'].median():.0f}")
print(f"   Mínimo:   {df_base['NUM_PERIODOS_CURSADOS'].min():.0f}")
print(f"   Máximo:   {df_base['NUM_PERIODOS_CURSADOS'].max():.0f}")

# -----------------------------------------------------------------------------
#Features temporales
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("FEATURES TEMPORALES")
print("-"*80)

features_temporales = []

# Año de ingreso
if 'AÑO' in df_base.columns:
    df_base['AÑO_INGRESO'] = df_base['AÑO'].astype(int)
    features_temporales.append('AÑO_INGRESO')
    print(f"   Feature creada: AÑO_INGRESO")

# Semestre de ingreso
if 'SEMESTRE' in df_base.columns:
    df_base['SEMESTRE_INGRESO'] = df_base['SEMESTRE'].astype(int)
    features_temporales.append('SEMESTRE_INGRESO')
    print(f"   Feature creada: SEMESTRE_INGRESO")

# Categoría de permanencia
def categorizar_permanencia(n_periodos):
    if n_periodos <= 2:
        return 'MUY_CORTA'
    elif n_periodos <= 4:
        return 'CORTA'
    elif n_periodos <= 6:
        return 'MEDIA'
    else:
        return 'LARGA'

df_base['CATEGORIA_PERMANENCIA'] = df_base['NUM_PERIODOS_CURSADOS'].apply(categorizar_permanencia)
features_temporales.append('CATEGORIA_PERMANENCIA')

print(f"   Feature creada: CATEGORIA_PERMANENCIA")
dist_perm = df_base['CATEGORIA_PERMANENCIA'].value_counts()
for cat, count in dist_perm.items():
    print(f"      {cat:15s}: {count:,} ({count/len(df_base)*100:.1f}%)")

print(f"\nTotal features temporales: {len(features_temporales)}")

# -----------------------------------------------------------------------------
#Feature Engineering - Variables económicas
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("FEATURE ENGINEERING - VARIABLES ECONÓMICAS")
print("-"*80)

features_economicas = []

# 1. Ratio Ingreso/Egreso
if all(col in df_base.columns for col in ['ESE_IGH_TOTAL_INGRESO', 'EGRESO']):
    df_base['RATIO_INGRESO_EGRESO'] = (
        df_base['ESE_IGH_TOTAL_INGRESO'] / (df_base['EGRESO'] + 1)
    ).replace([np.inf, -np.inf], np.nan)
    features_economicas.append('RATIO_INGRESO_EGRESO')
    print(f"   1. RATIO_INGRESO_EGRESO")

# 2. Balance económico
if all(col in df_base.columns for col in ['ESE_IGH_TOTAL_INGRESO', 'EGRESO']):
    df_base['BALANCE_ECONOMICO'] = df_base['ESE_IGH_TOTAL_INGRESO'] - df_base['EGRESO']
    features_economicas.append('BALANCE_ECONOMICO')
    print(f"   2. BALANCE_ECONOMICO")

# 3. Vulnerabilidad económica
if 'BALANCE_ECONOMICO' in df_base.columns:
    df_base['VULNERABLE_ECONOMICO'] = (df_base['BALANCE_ECONOMICO'] < 0).astype(int)
    features_economicas.append('VULNERABLE_ECONOMICO')
    print(f"   3. VULNERABLE_ECONOMICO")

# 4. Presión económica en educación
if all(col in df_base.columns for col in ['EGRESO_EDUCACION', 'ESE_IGH_TOTAL_INGRESO']):
    df_base['PRESION_EDUCACION'] = (
        df_base['EGRESO_EDUCACION'] / (df_base['ESE_IGH_TOTAL_INGRESO'] + 1)
    ).replace([np.inf, -np.inf], np.nan)
    features_economicas.append('PRESION_EDUCACION')
    print(f"   4. PRESION_EDUCACION")

# 5. Presión vivienda
if all(col in df_base.columns for col in ['EGRESO_VIVIENDA', 'ESE_IGH_TOTAL_INGRESO']):
    df_base['PRESION_VIVIENDA'] = (
        df_base['EGRESO_VIVIENDA'] / (df_base['ESE_IGH_TOTAL_INGRESO'] + 1)
    ).replace([np.inf, -np.inf], np.nan)
    features_economicas.append('PRESION_VIVIENDA')
    print(f"   5. PRESION_VIVIENDA")

# 6. Presión préstamos
if all(col in df_base.columns for col in ['EGRESO_PRESTAMOS', 'ESE_IGH_TOTAL_INGRESO']):
    df_base['PRESION_PRESTAMOS'] = (
        df_base['EGRESO_PRESTAMOS'] / (df_base['ESE_IGH_TOTAL_INGRESO'] + 1)
    ).replace([np.inf, -np.inf], np.nan)
    features_economicas.append('PRESION_PRESTAMOS')
    print(f"   6. PRESION_PRESTAMOS")

# 7. Egreso promedio por período
if all(col in df_base.columns for col in ['EGRESO', 'NUM_PERIODOS_CURSADOS']):
    df_base['EGRESO_PROMEDIO_PERIODO'] = (
        df_base['EGRESO'] / (df_base['NUM_PERIODOS_CURSADOS'] + 1)
    )
    features_economicas.append('EGRESO_PROMEDIO_PERIODO')
    print(f"   7. EGRESO_PROMEDIO_PERIODO")

# 8. Tiene hijos
if 'HIJOS' in df_base.columns:
    df_base['TIENE_HIJOS'] = (df_base['HIJOS'] > 0).astype(int)
    features_economicas.append('TIENE_HIJOS')
    print(f"   8. TIENE_HIJOS")

print(f"\nTotal features económicas: {len(features_economicas)}")

# -----------------------------------------------------------------------------
#Identificar features disponibles
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("IDENTIFICACIÓN DE FEATURES")
print("-"*80)

# Columnas a excluir
excluir = [
    'ESTUDIANTE', 'ESTADO_FINAL', 'DESERCION', 'DESERCION_ESTADO',
    'DESERCION_DESAPARICION', 'DESERCION_TEMPRANA',
    'PERIODO_ORIGINAL', 'PERIODO_LABEL', 'N'
]

# Excluir variables MISSING_*
excluir.extend([col for col in df_base.columns if col.startswith('MISSING_')])

print(f"\nColumnas excluidas: {len(excluir)}")

todas_features = [col for col in df_base.columns if col not in excluir]
print(f"Features potenciales: {len(todas_features)}")

# -----------------------------------------------------------------------------
#Análisis de completitud
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("ANÁLISIS DE COMPLETITUD")
print("-"*80)

completitud = (1 - df_base[todas_features].isnull().sum() / len(df_base)) * 100

UMBRAL_COMPLETITUD = 70
features_validas = completitud[completitud > UMBRAL_COMPLETITUD].index.tolist()
features_descartadas = completitud[completitud <= UMBRAL_COMPLETITUD].index.tolist()

print(f"\nUmbral: {UMBRAL_COMPLETITUD}%")
print(f"   Válidas: {len(features_validas)}")
print(f"   Descartadas: {len(features_descartadas)}")

todas_features = features_validas

# -----------------------------------------------------------------------------
#Clasificación por tipo
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("CLASIFICACIÓN POR TIPO")
print("-"*80)

cat_cols = []
num_cols = []

variables_forzar_numericas = ['AÑO', 'SEMESTRE', 'AÑO_INGRESO', 'SEMESTRE_INGRESO', 'NUM_PERIODOS_CURSADOS']

for var in variables_forzar_numericas:
    if var in todas_features:
        num_cols.append(var)
        todas_features.remove(var)

for col in todas_features:
    dtype = df_base[col].dtype
    n_unique = df_base[col].nunique()
    
    if dtype == 'object':
        cat_cols.append(col)
    elif dtype in ['int64', 'float64']:
        if n_unique <= 10:
            cat_cols.append(col)
        else:
            num_cols.append(col)

print(f"\nResultado:")
print(f"   Categóricas: {len(cat_cols)}")
print(f"   Numéricas:   {len(num_cols)}")
print(f"   Total:       {len(cat_cols) + len(num_cols)}")

# -----------------------------------------------------------------------------
# Preparar X e y
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("PREPARACIÓN DE X E Y")
print("-"*80)

X = df_base[cat_cols + num_cols].copy()
y = df_base['DESERCION'].copy()

print(f"\nDimensiones:")
print(f"   X: {X.shape} ({X.shape[0]:,} x {X.shape[1]})")
print(f"   y: {y.shape}")
print(f"\nComposición:")
print(f"   Categóricas: {len(cat_cols)}")
print(f"   Numéricas:   {len(num_cols)}")

# -----------------------------------------------------------------------------
# División Train/Test
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("DIVISIÓN TRAIN/TEST")
print("-"*80)

print(f"\nConfiguración:")
print(f"   Train: {(1-TEST_SIZE)*100:.0f}%")
print(f"   Test:  {TEST_SIZE*100:.0f}%")
print(f"   Estratificación: Activada")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

print(f"\nTRAIN:")
print(f"   Registros: {len(X_train):,}")
print(f"   Desertores: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")

print(f"\nTEST:")
print(f"   Registros: {len(X_test):,}")
print(f"   Desertores: {y_test.sum():,} ({y_test.mean()*100:.2f}%)")

diferencia = abs(y_train.mean() - y_test.mean())
print(f"\nEstratificación: {diferencia:.6f} ({'EXCELENTE' if diferencia < 0.001 else 'BUENA'})")

# -----------------------------------------------------------------------------
#Pipeline de preprocesamiento
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("PIPELINE DE PREPROCESAMIENTO")
print("-"*80)

print(f"\nTransformador CATEGÓRICO:")
print(f"   1. Imputer: DESCONOCIDO")
print(f"   2. OneHotEncoder: max 20 categorías")

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='DESCONOCIDO')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=20))
])

print(f"\nTransformador NUMÉRICO:")
print(f"   1. Imputer: mediana")
print(f"   2. StandardScaler")

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', cat_transformer, cat_cols),
        ('num', num_transformer, num_cols)
    ],
    remainder='drop'
)

print(f"\nPipeline configurado:")
print(f"   Categóricas: {len(cat_cols)}")
print(f"   Numéricas:   {len(num_cols)}")

# -----------------------------------------------------------------------------
#Sample weights
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("SAMPLE WEIGHTS")
print("-"*80)

sample_weights_train = compute_sample_weight(class_weight='balanced', y=y_train)
sample_weights_test = compute_sample_weight(class_weight='balanced', y=y_test)

print(f"\nWeights calculados:")
print(f"   Train: min={sample_weights_train.min():.4f}, max={sample_weights_train.max():.4f}")
print(f"   Test:  min={sample_weights_test.min():.4f}, max={sample_weights_test.max():.4f}")

# -----------------------------------------------------------------------------
#Guardar artefactos
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("GUARDANDO ARTEFACTOS")
print("-"*80)

config_features = {
    'cat_cols': cat_cols,
    'num_cols': num_cols,
    'features_economicas': features_economicas,
    'features_temporales': features_temporales,
    'n_features_total': len(cat_cols) + len(num_cols)
}

with open(DIRS['resultados'] / 'config_features.pkl', 'wb') as f:
    pickle.dump(config_features, f)

with open(DIRS['modelos'] / 'preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

splits_data = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
    'sample_weights_train': sample_weights_train,
    'sample_weights_test': sample_weights_test
}

with open(DIRS['datos'] / 'train_test_splits.pkl', 'wb') as f:
    pickle.dump(splits_data, f)

print(f"\nArtefactos guardados:")
print(f"   - config_features.pkl")
print(f"   - preprocessor.pkl")
print(f"   - train_test_splits.pkl")

# -----------------------------------------------------------------------------
# Visualización
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("GENERANDO VISUALIZACIÓN")
print("-"*80)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Preparación para Modelado', fontsize=16, fontweight='bold')

# Gráfico 1: Composición
axes[0].pie([len(cat_cols), len(num_cols)], 
           labels=[f'Categóricas\n{len(cat_cols)}', f'Numéricas\n{len(num_cols)}'],
           colors=['#e74c3c', '#3498db'], autopct='%1.1f%%',
           textprops={'fontsize': 12, 'fontweight': 'bold'}, startangle=90)
axes[0].set_title('Composición de Features', fontweight='bold', fontsize=13)

# Gráfico 2: Balance Train/Test
train_no = (~y_train.astype(bool)).sum()
train_si = y_train.sum()
test_no = (~y_test.astype(bool)).sum()
test_si = y_test.sum()

x = np.arange(2)
width = 0.35

axes[1].bar(x - width/2, [train_no, test_no], width, label='No Desertores', 
           color='#2ecc71', edgecolor='black', alpha=0.7)
axes[1].bar(x + width/2, [train_si, test_si], width, label='Desertores', 
           color='#e74c3c', edgecolor='black', alpha=0.7)

axes[1].set_ylabel('Cantidad', fontweight='bold', fontsize=12)
axes[1].set_title('Balance Train/Test', fontweight='bold', fontsize=13)
axes[1].set_xticks(x)
axes[1].set_xticklabels(['Train', 'Test'])
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(DIRS['visualizaciones'] / 'preparacion_modelado.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"   Visualización guardada: preparacion_modelado.png")

# -----------------------------------------------------------------------------
# Resumen
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("RESUMEN - BLOQUE COMPLETADO")
print("="*80)

print(f"""
DATOS PREPARADOS PARA PREDICCIÓN DE DESERCIÓN

FEATURES:
   Total: {X.shape[1]}
   Económicas: {len(features_economicas)}
   Temporales: {len(features_temporales)}

DIVISIÓN:
   Train: {len(X_train):,}
   Test:  {len(X_test):,}

BALANCE:
   Deserción: {y_train.mean()*100:.2f}%
   Ratio: {train_no/train_si:.2f}:1

ARCHIVOS GENERADOS:
   - config_features.pkl
   - preprocessor.pkl
   - train_test_splits.pkl
   - preparacion_modelado.png

PRÓXIMO PASO:
   Ejecutar src/04_decision_tree.py (BLOQUE 5A)
""")

print("="*80)
print("COMPLETADO EXITOSAMENTE")
print("="*80)
"""
CREACIÓN DE VARIABLE DESERCIÓN (VERSIÓN CORREGIDA)
Implementación de definición multi-criterio con validación de retornos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import sys

warnings.filterwarnings('ignore')

# Importar configuración
sys.path.insert(0, str(Path(__file__).parent))
exec(open(Path(__file__).parent / '00_config.py').read())

print("="*80)
print("CREACIÓN DE VARIABLE DESERCIÓN (VERSIÓN CORREGIDA)")
print("="*80)

# -----------------------------------------------------------------------------
#Cargar dataset
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("CARGANDO DATASET")
print("-"*80)

ruta_dataset = DIRS['datos'] / 'dataset_normalizado.csv'
print(f"\nCargando desde: {ruta_dataset}")

df = pd.read_csv(ruta_dataset)

print(f"\nDataset cargado:")
print(f"   Registros: {len(df):,}")
print(f"   Columnas: {len(df.columns)}")

# Verificar columnas necesarias
columnas_necesarias = ['ESTUDIANTE', 'ESTADO_FINAL', 'PERIODO_LABEL']
for col in columnas_necesarias:
    if col not in df.columns:
        raise ValueError(f"ERROR: Columna requerida '{col}' no encontrada")

print("\nColumnas necesarias verificadas correctamente")

# -----------------------------------------------------------------------------
# Ordenar datos
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("PREPARACIÓN DE DATOS LONGITUDINALES")
print("-"*80)

print("\nOrdenando datos por estudiante y período cronológico...")
df = df.sort_values(['ESTUDIANTE', 'PERIODO_LABEL']).reset_index(drop=True)
print("   Datos ordenados correctamente")

print("\nCreando estructura de análisis longitudinal...")
estudiantes_info = {}

for estudiante_id in df['ESTUDIANTE'].unique():
    df_estudiante = df[df['ESTUDIANTE'] == estudiante_id].copy()
    
    estudiantes_info[estudiante_id] = {
        'periodos': df_estudiante['PERIODO_LABEL'].tolist(),
        'n_periodos': len(df_estudiante),
        'estados': df_estudiante['ESTADO_FINAL'].tolist(),
        'estado_final': df_estudiante['ESTADO_FINAL'].iloc[-1],
        'indices': df_estudiante.index.tolist()
    }

print(f"   Estructura creada para {len(estudiantes_info):,} estudiantes")

# -----------------------------------------------------------------------------
#Identificar períodos académicos
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("IDENTIFICACIÓN DE PERÍODOS ACADÉMICOS")
print("-"*80)

periodos_ordenados = sorted(df['PERIODO_LABEL'].unique())
n_periodos_totales = len(periodos_ordenados)

print(f"\nPeríodos académicos en el dataset: {n_periodos_totales}")
print("\nLista de períodos (orden cronológico):")
for i, periodo in enumerate(periodos_ordenados, 1):
    n_registros = len(df[df['PERIODO_LABEL'] == periodo])
    n_estudiantes_periodo = df[df['PERIODO_LABEL'] == periodo]['ESTUDIANTE'].nunique()
    print(f"   {i:2d}. {periodo:15s}: {n_registros:6,} registros | {n_estudiantes_periodo:6,} estudiantes")

ultimos_2_periodos = periodos_ordenados[-2:]
print(f"\nÚltimos 2 períodos identificados para análisis de desaparición:")
for periodo in ultimos_2_periodos:
    print(f"   - {periodo}")

# Crear índice de períodos para validaciones
periodo_a_indice = {periodo: idx for idx, periodo in enumerate(periodos_ordenados)}
print(f"\nÍndice de períodos creado para validaciones de continuidad")

# -----------------------------------------------------------------------------
#Inicializar variables de deserción
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("INICIALIZACIÓN DE VARIABLES")
print("-"*80)

df['DESERCION'] = 0
df['DESERCION_ESTADO'] = 0
df['DESERCION_DESAPARICION'] = 0
df['DESERCION_TEMPRANA'] = 0

print("\nVariables de deserción inicializadas:")
print("   - DESERCION: Variable principal (0 = no desertor, 1 = desertor)")
print("   - DESERCION_ESTADO: Deserción por estado final")
print("   - DESERCION_DESAPARICION: Deserción por desaparición REAL")
print("   - DESERCION_TEMPRANA: Deserción temprana (1-2 períodos)")

# -----------------------------------------------------------------------------
# Aplicar CRITERIO 1 - Deserción por ESTADO_FINAL
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("APLICANDO CRITERIO 1 - DESERCIÓN POR ESTADO_FINAL")
print("-"*80)

valores_desercion_estado = ['RETIRADO', 'BAJA', 'RETIRO', 'DESERTOR', 'ABANDONO']

print("\nBuscando estudiantes con estados que indican deserción...")

mask_estado = df['ESTADO_FINAL'].str.strip().str.upper().isin(valores_desercion_estado)
df.loc[mask_estado, 'DESERCION_ESTADO'] = 1
df.loc[mask_estado, 'DESERCION'] = 1

n_desercion_estado = mask_estado.sum()
estudiantes_desercion_estado = df[mask_estado]['ESTUDIANTE'].nunique()

print(f"\nResultados CRITERIO 1:")
print(f"   Registros marcados: {n_desercion_estado:,}")
print(f"   Estudiantes afectados: {estudiantes_desercion_estado:,}")

# -----------------------------------------------------------------------------
#Aplicar CRITERIO 2 - Deserción por desaparición (CORREGIDO)
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("APLICANDO CRITERIO 2 - DESERCIÓN POR DESAPARICIÓN (CORREGIDO)")
print("-"*80)

print("\nAnalizando estudiantes que desaparecen y NO vuelven...")
print(f"   Validaciones aplicadas:")
print(f"   1. Cursó 2 o más períodos")
print(f"   2. NO aparece en los últimos 2 períodos: {ultimos_2_periodos}")
print(f"   3. Han pasado 2+ períodos desde última aparición")
print(f"   4. NO retornó después de ausencia (clave para evitar falsos positivos)")

contador_desaparicion = 0
estudiantes_desaparicion = []
estudiantes_que_retornaron = []

for estudiante_id, info in estudiantes_info.items():
    if info['n_periodos'] >= 2:
        aparece_ultimos_2 = any(periodo in ultimos_2_periodos for periodo in info['periodos'])
        
        if not aparece_ultimos_2:
            # VALIDACIÓN CRÍTICA: Verificar que realmente desertó y NO volvió
            indices_periodos_cursados = sorted([periodo_a_indice[p] for p in info['periodos']])
            ultimo_periodo_cursado_idx = indices_periodos_cursados[-1]
            indice_ultimo_periodo_disponible = len(periodos_ordenados) - 1
            periodos_ausente = indice_ultimo_periodo_disponible - ultimo_periodo_cursado_idx
            
            # Validar si tiene patrón de retorno
            tiene_retorno = False
            for i in range(len(indices_periodos_cursados) - 1):
                gap = indices_periodos_cursados[i+1] - indices_periodos_cursados[i]
                if gap > 1:
                    tiene_retorno = True
                    break
            
            if periodos_ausente >= 2 and not tiene_retorno:
                indices = info['indices']
                df.loc[indices, 'DESERCION_DESAPARICION'] = 1
                df.loc[indices, 'DESERCION'] = 1
                contador_desaparicion += len(indices)
                estudiantes_desaparicion.append(estudiante_id)
            elif tiene_retorno:
                estudiantes_que_retornaron.append(estudiante_id)

print(f"\nResultados CRITERIO 2 (CORREGIDO):")
print(f"   Registros marcados como deserción: {contador_desaparicion:,}")
print(f"   Estudiantes marcados como desertores: {len(estudiantes_desaparicion):,}")

print(f"\nValidación de corrección:")
print(f"   Estudiantes que retornaron después de ausencias: {len(estudiantes_que_retornaron):,}")
print(f"   Estos NO fueron marcados como desertores (correcto)")

# -----------------------------------------------------------------------------
#Aplicar CRITERIO 3 - Deserción temprana
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("APLICANDO CRITERIO 3 - DESERCIÓN TEMPRANA")
print("-"*80)

print("\nAnalizando estudiantes que solo cursaron 1-2 períodos iniciales...")

contador_temprana = 0
estudiantes_temprana = []

for estudiante_id, info in estudiantes_info.items():
    if info['n_periodos'] <= 2:
        indices = info['indices']
        ya_marcado = df.loc[indices[0], 'DESERCION'] == 1
        
        if not ya_marcado:
            df.loc[indices, 'DESERCION_TEMPRANA'] = 1
            df.loc[indices, 'DESERCION'] = 1
            contador_temprana += len(indices)
            estudiantes_temprana.append(estudiante_id)

print(f"\nResultados CRITERIO 3:")
print(f"   Registros marcados: {contador_temprana:,}")
print(f"   Estudiantes afectados: {len(estudiantes_temprana):,}")

# -----------------------------------------------------------------------------
#Crear dataset base
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("CREACIÓN DE DATASET BASE")
print("-"*80)

print("\nCreando dataset con un registro por estudiante...")
print("   Estrategia: Último registro de cada estudiante")

df_base = df.groupby('ESTUDIANTE').last().reset_index()

print(f"\nDataset base creado:")
print(f"   Registros originales: {len(df):,}")
print(f"   Registros en df_base: {len(df_base):,}")
print(f"   Columnas: {len(df_base.columns)}")

desertores_base = df_base['DESERCION'].sum()
no_desertores_base = len(df_base) - desertores_base

print(f"\nDistribución en df_base:")
print(f"   No desertores: {no_desertores_base:,} ({no_desertores_base/len(df_base)*100:.2f}%)")
print(f"   Desertores:    {desertores_base:,} ({desertores_base/len(df_base)*100:.2f}%)")

# -----------------------------------------------------------------------------
# Visualización
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("GENERANDO VISUALIZACIÓN")
print("-"*80)

print("\nGenerando visualización de deserción...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Análisis de Deserción - Variable Corregida', fontsize=16, fontweight='bold')

# Gráfico 1: Deserción a nivel registro
n_desertores = df['DESERCION'].sum()
n_no_desertores = len(df) - n_desertores

axes[0, 0].bar(['No Desertor', 'Desertor'], [n_no_desertores, n_desertores],
               color=['#2ecc71', '#e74c3c'], edgecolor='black', alpha=0.8, width=0.6)
axes[0, 0].set_ylabel('Cantidad de Registros', fontweight='bold', fontsize=12)
axes[0, 0].set_title('Deserción a Nivel Registro', fontweight='bold', fontsize=13)
axes[0, 0].grid(axis='y', alpha=0.3)

for i, v in enumerate([n_no_desertores, n_desertores]):
    axes[0, 0].text(i, v, f'{v:,}\n({v/len(df)*100:.1f}%)', 
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

# Gráfico 2: Deserción a nivel estudiante
axes[0, 1].bar(['No Desertor', 'Desertor'], [no_desertores_base, desertores_base],
               color=['#2ecc71', '#e74c3c'], edgecolor='black', alpha=0.8, width=0.6)
axes[0, 1].set_ylabel('Cantidad de Estudiantes', fontweight='bold', fontsize=12)
axes[0, 1].set_title('Deserción a Nivel Estudiante', fontweight='bold', fontsize=13)
axes[0, 1].grid(axis='y', alpha=0.3)

for i, v in enumerate([no_desertores_base, desertores_base]):
    axes[0, 1].text(i, v, f'{v:,}\n({v/len(df_base)*100:.1f}%)', 
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

# Gráfico 3: Por criterio
criterios = ['Estado', 'Desaparición', 'Temprana']
valores_est = [estudiantes_desercion_estado, len(estudiantes_desaparicion), len(estudiantes_temprana)]
colores_crit = ['#e74c3c', '#f39c12', '#9b59b6']

axes[1, 0].bar(criterios, valores_est, color=colores_crit, edgecolor='black', alpha=0.8)
axes[1, 0].set_ylabel('Estudiantes', fontweight='bold', fontsize=12)
axes[1, 0].set_title('Desertores por Criterio', fontweight='bold', fontsize=13)
axes[1, 0].grid(axis='y', alpha=0.3)

for i, v in enumerate(valores_est):
    axes[1, 0].text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Gráfico 4: Pie chart
axes[1, 1].pie([no_desertores_base, desertores_base], 
               labels=[f'No Desertor\n{no_desertores_base:,}', f'Desertor\n{desertores_base:,}'],
               colors=['#2ecc71', '#e74c3c'], autopct='%1.1f%%',
               explode=(0, 0.1), startangle=90,
               textprops={'fontsize': 11, 'fontweight': 'bold'})
axes[1, 1].set_title('Proporción Final', fontweight='bold', fontsize=13)

plt.tight_layout()
plt.savefig(DIRS['visualizaciones'] / 'desercion_analisis_corregido.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"   Visualización guardada: desercion_analisis_corregido.png")

# -----------------------------------------------------------------------------
# Guardar resultados
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("GUARDANDO RESULTADOS")
print("-"*80)

# Guardar dataset completo
ruta_df_completo = DIRS['datos'] / 'dataset_con_desercion_corregido.csv'
df.to_csv(ruta_df_completo, index=False)
print(f"\nDataset completo guardado: dataset_con_desercion_corregido.csv")

# Guardar dataset base
ruta_df_base = DIRS['datos'] / 'dataset_base_modelado_corregido.csv'
df_base.to_csv(ruta_df_base, index=False)
print(f"Dataset base guardado: dataset_base_modelado_corregido.csv")

# Guardar métricas
metricas_desercion = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'version': 'corregida_con_validacion_retornos',
    'n_registros_total': int(len(df)),
    'n_estudiantes_total': int(len(df_base)),
    'n_estudiantes_desertores': int(desertores_base),
    'n_estudiantes_que_retornaron': int(len(estudiantes_que_retornaron)),
    'pct_desercion_estudiantil': float(desertores_base/len(df_base)*100),
    'desertores_por_estado': int(estudiantes_desercion_estado),
    'desertores_por_desaparicion_sin_retorno': int(len(estudiantes_desaparicion)),
    'desertores_por_temprana': int(len(estudiantes_temprana))
}

guardar_resultados_json(metricas_desercion, 'metricas_desercion_corregida')

# -----------------------------------------------------------------------------
# Resumen
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("RESUMEN - BLOQUE 3 COMPLETADO")
print("="*80)

print(f"""
VARIABLE DESERCIÓN CREADA CON VALIDACIÓN DE RETORNOS

CORRECCIÓN IMPLEMENTADA:
   Criterio 2 ahora detecta y excluye estudiantes que retornaron
   Estudiantes con retornos detectados: {len(estudiantes_que_retornaron):,}
   Estos NO fueron marcados como desertores (correcto)

CRITERIOS FINALES:
   1. Estado final: {estudiantes_desercion_estado:,} estudiantes
   2. Desaparición SIN retorno: {len(estudiantes_desaparicion):,} estudiantes
   3. Temprana: {len(estudiantes_temprana):,} estudiantes

RESULTADOS:
   Desertores: {desertores_base:,} ({desertores_base/len(df_base)*100:.2f}%)
   Ratio: {no_desertores_base/desertores_base:.2f}:1

DATASET BASE:
   Registros: {len(df_base):,}
   Desertores: {desertores_base:,}

ARCHIVOS GENERADOS:
   - dataset_con_desercion_corregido.csv
   - dataset_base_modelado_corregido.csv
   - metricas_desercion_corregida.json
   - desercion_analisis_corregido.png

PRÓXIMO PASO:
   Ejecutar src/03_feature_engineering.py 
""")

print("="*80)
print("COMPLETADO EXITOSAMENTE")
print("="*80)
"""
BLOQUE 8: COMPARACIÓN FINAL DE TODOS LOS MODELOS
Análisis comparativo exhaustivo de los 5 modelos
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import sys
import json
import pickle

warnings.filterwarnings('ignore')

# Importar configuración
sys.path.insert(0, str(Path(__file__).parent))
exec(open(Path(__file__).parent / '00_config.py').read())

print("="*80)
print("COMPARACIÓN FINAL DE MODELOS")
print("Análisis comparativo de los 5 modelos entrenados")
print("="*80)

# -----------------------------------------------------------------------------
# Cargar métricas de todos los modelos
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("CARGANDO MÉTRICAS")
print("-"*80)

modelos = {
    'Decision Tree': 'metricas_decision_tree.json',
    'Random Forest': 'metricas_random_forest.json',
    'Gradient Boosting': 'metricas_gradient_boosting.json',
    'Regresion Logistica': 'metricas_regresion_logistica.json',
    'XGBoost': 'metricas_xgboost.json'
}

metricas_todos = {}

print("\nCargando métricas de modelos:")
for nombre, archivo in modelos.items():
    ruta = DIRS['metricas'] / archivo
    try:
        with open(ruta, 'r') as f:
            metricas_todos[nombre] = json.load(f)
        print(f"   ✓ {nombre}")
    except FileNotFoundError:
        print(f"   ✗ {nombre} - Archivo no encontrado")

print(f"\nModelos cargados: {len(metricas_todos)}")

# -----------------------------------------------------------------------------
# Tabla comparativa
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("TABLA COMPARATIVA DE MÉTRICAS")
print("-"*80)

print(f"\n{'='*120}")
print(f"COMPARACIÓN EXHAUSTIVA DE MODELOS - CONJUNTO DE PRUEBA")
print(f"{'='*120}")

print(f"\n{'Métrica':25s} {'Decision Tree':>18s} {'Random Forest':>18s} {'Gradient Boost':>18s} {'Reg. Logística':>18s} {'XGBoost':>18s}")
print("-" * 120)

# Extraer métricas
metricas_comparar = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

for metrica in metricas_comparar:
    valores = []
    for modelo in ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'Regresion Logistica', 'XGBoost']:
        if modelo in metricas_todos:
            valor = metricas_todos[modelo]['metricas_test'].get(metrica, 0.0)
            valores.append(f"{valor:>18.4f}")
        else:
            valores.append(f"{'N/A':>18s}")
    
    print(f"{metrica.upper():25s} {''.join(valores)}")

print("\n" + "-" * 120)
print("MATRIZ DE CONFUSIÓN")
print("-" * 120)

# Falsos Negativos
print(f"{'Falsos Negativos (FN)':25s}", end='')
for modelo in ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'Regresion Logistica', 'XGBoost']:
    if modelo in metricas_todos:
        fn = metricas_todos[modelo]['matriz_confusion']['FN']
        print(f"{fn:>18,}", end='')
    else:
        print(f"{'N/A':>18s}", end='')
print()

# Verdaderos Positivos
print(f"{'Verdaderos Positivos (TP)':25s}", end='')
for modelo in ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'Regresion Logistica', 'XGBoost']:
    if modelo in metricas_todos:
        tp = metricas_todos[modelo]['matriz_confusion']['TP']
        print(f"{tp:>18,}", end='')
    else:
        print(f"{'N/A':>18s}", end='')
print()

print("\n" + "-" * 120)
print("INFORMACIÓN DEL ENTRENAMIENTO")
print("-" * 120)

# Tiempo
print(f"{'Tiempo (minutos)':25s}", end='')
for modelo in ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'Regresion Logistica', 'XGBoost']:
    if modelo in metricas_todos:
        tiempo = metricas_todos[modelo].get('tiempo_minutos', 0.0)
        print(f"{tiempo:>18.2f}", end='')
    else:
        print(f"{'N/A':>18s}", end='')
print()

# CV Score
print(f"{'CV F1-Score':25s}", end='')
for modelo in ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'Regresion Logistica', 'XGBoost']:
    if modelo in metricas_todos:
        cv = metricas_todos[modelo].get('cv_score', 0.0)
        print(f"{cv:>18.4f}", end='')
    else:
        print(f"{'N/A':>18s}", end='')
print()

print("\n" + "="*120)

# -----------------------------------------------------------------------------
# Identificar mejor modelo
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("IDENTIFICACIÓN DEL MEJOR MODELO")
print("-"*80)

print("\nMEJOR MODELO POR MÉTRICA:")
print(f"\n{'Métrica':20s} {'Mejor Modelo':25s} {'Valor':>15s}")
print("-" * 65)

for metrica in metricas_comparar:
    mejor_modelo = None
    mejor_valor = -1
    
    for modelo in metricas_todos:
        valor = metricas_todos[modelo]['metricas_test'].get(metrica, 0.0)
        if valor > mejor_valor:
            mejor_valor = valor
            mejor_modelo = modelo
    
    print(f"{metrica.upper():20s} {mejor_modelo:25s} {mejor_valor:>15.4f}")

# Mejor modelo por F1-Score
f1_scores = {}
for modelo in metricas_todos:
    f1_scores[modelo] = metricas_todos[modelo]['metricas_test']['f1_score']

mejor_modelo_global = max(f1_scores, key=f1_scores.get)
mejor_f1 = f1_scores[mejor_modelo_global]

print(f"\n{'='*65}")
print(f"MODELO GANADOR (F1-Score): {mejor_modelo_global}")
print(f"F1-Score: {mejor_f1:.4f}")
print(f"{'='*65}")

# Mejor modelo por Falsos Negativos
fn_counts = {}
for modelo in metricas_todos:
    fn_counts[modelo] = metricas_todos[modelo]['matriz_confusion']['FN']

mejor_fn = min(fn_counts, key=fn_counts.get)
menor_fn = fn_counts[mejor_fn]

print(f"\nMENOR FALSOS NEGATIVOS (CRÍTICO):")
print(f"   Modelo: {mejor_fn}")
print(f"   FN: {menor_fn:,} desertores NO detectados")

# -----------------------------------------------------------------------------
# Paso 4: Visualizaciones comparativas
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("PASO 4: GENERANDO VISUALIZACIONES COMPARATIVAS")
print("-"*80)

# VISUALIZACIÓN 1: Comparación de métricas
print("\nVisualizando comparación de métricas...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Comparación Final de Modelos - Todas las Métricas', fontsize=18, fontweight='bold')

# Gráfico 1: Métricas principales
modelos_list = list(metricas_todos.keys())
metricas_principales = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

valores_dt = []
valores_rf = []
valores_gb = []
valores_lr = []
valores_xgb = []

for metrica in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
    if 'Decision Tree' in metricas_todos:
        valores_dt.append(metricas_todos['Decision Tree']['metricas_test'][metrica])
    if 'Random Forest' in metricas_todos:
        valores_rf.append(metricas_todos['Random Forest']['metricas_test'][metrica])
    if 'Gradient Boosting' in metricas_todos:
        valores_gb.append(metricas_todos['Gradient Boosting']['metricas_test'][metrica])
    if 'Regresion Logistica' in metricas_todos:
        valores_lr.append(metricas_todos['Regresion Logistica']['metricas_test'][metrica])
    if 'XGBoost' in metricas_todos:
        valores_xgb.append(metricas_todos['XGBoost']['metricas_test'][metrica])

x = np.arange(len(metricas_principales))
width = 0.15

if valores_dt:
    axes[0, 0].bar(x - 2*width, valores_dt, width, label='Decision Tree', color='#3498db', edgecolor='black', alpha=0.8)
if valores_rf:
    axes[0, 0].bar(x - width, valores_rf, width, label='Random Forest', color='#2ecc71', edgecolor='black', alpha=0.8)
if valores_gb:
    axes[0, 0].bar(x, valores_gb, width, label='Gradient Boosting', color='#e74c3c', edgecolor='black', alpha=0.8)
if valores_lr:
    axes[0, 0].bar(x + width, valores_lr, width, label='Reg. Logística', color='#f39c12', edgecolor='black', alpha=0.8)
if valores_xgb:
    axes[0, 0].bar(x + 2*width, valores_xgb, width, label='XGBoost', color='#9b59b6', edgecolor='black', alpha=0.8)

axes[0, 0].set_ylabel('Score', fontweight='bold', fontsize=12)
axes[0, 0].set_title('Comparación de Métricas Principales', fontweight='bold', fontsize=14)
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(metricas_principales, fontsize=11)
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(axis='y', alpha=0.3)
axes[0, 0].set_ylim([0, 1.1])

# Gráfico 2: F1-Score destacado
modelos_nombres = []
f1_valores = []
colores_modelos = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

for i, modelo in enumerate(['Decision Tree', 'Random Forest', 'Gradient Boosting', 'Regresion Logistica', 'XGBoost']):
    if modelo in metricas_todos:
        modelos_nombres.append(modelo.replace(' ', '\n'))
        f1_valores.append(metricas_todos[modelo]['metricas_test']['f1_score'])

barras = axes[0, 1].bar(range(len(f1_valores)), f1_valores, color=colores_modelos[:len(f1_valores)], 
                        edgecolor='black', alpha=0.8, width=0.6)

axes[0, 1].set_ylabel('F1-Score', fontweight='bold', fontsize=12)
axes[0, 1].set_title('F1-Score - Métrica Principal', fontweight='bold', fontsize=14)
axes[0, 1].set_xticks(range(len(modelos_nombres)))
axes[0, 1].set_xticklabels(modelos_nombres, fontsize=10)
axes[0, 1].grid(axis='y', alpha=0.3)
axes[0, 1].set_ylim([0, 1.1])

# Marcar el mejor
if f1_valores:
    mejor_idx = f1_valores.index(max(f1_valores))
    barras[mejor_idx].set_edgecolor('gold')
    barras[mejor_idx].set_linewidth(4)

for i, (barra, valor) in enumerate(zip(barras, f1_valores)):
    axes[0, 1].text(barra.get_x() + barra.get_width()/2., valor + 0.02,
                   f'{valor:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Gráfico 3: Falsos Negativos
fn_valores = []
for modelo in ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'Regresion Logistica', 'XGBoost']:
    if modelo in metricas_todos:
        fn_valores.append(metricas_todos[modelo]['matriz_confusion']['FN'])

barras_fn = axes[1, 0].bar(range(len(fn_valores)), fn_valores, color=colores_modelos[:len(fn_valores)], 
                           edgecolor='black', alpha=0.8, width=0.6)

axes[1, 0].set_ylabel('Falsos Negativos', fontweight='bold', fontsize=12)
axes[1, 0].set_title('Falsos Negativos - Desertores NO Detectados (Menor es mejor)', 
                     fontweight='bold', fontsize=14)
axes[1, 0].set_xticks(range(len(modelos_nombres)))
axes[1, 0].set_xticklabels(modelos_nombres, fontsize=10)
axes[1, 0].grid(axis='y', alpha=0.3)

# Marcar el mejor (menor)
if fn_valores:
    mejor_fn_idx = fn_valores.index(min(fn_valores))
    barras_fn[mejor_fn_idx].set_edgecolor('gold')
    barras_fn[mejor_fn_idx].set_linewidth(4)

for i, (barra, valor) in enumerate(zip(barras_fn, fn_valores)):
    axes[1, 0].text(barra.get_x() + barra.get_width()/2., valor + max(fn_valores)*0.02,
                   f'{valor:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Gráfico 4: Tiempo de entrenamiento
tiempos = []
for modelo in ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'Regresion Logistica', 'XGBoost']:
    if modelo in metricas_todos:
        tiempos.append(metricas_todos[modelo].get('tiempo_minutos', 0.0))

axes[1, 1].bar(range(len(tiempos)), tiempos, color=colores_modelos[:len(tiempos)], 
               edgecolor='black', alpha=0.8, width=0.6)

axes[1, 1].set_ylabel('Tiempo (minutos)', fontweight='bold', fontsize=12)
axes[1, 1].set_title('Tiempo de Entrenamiento', fontweight='bold', fontsize=14)
axes[1, 1].set_xticks(range(len(modelos_nombres)))
axes[1, 1].set_xticklabels(modelos_nombres, fontsize=10)
axes[1, 1].grid(axis='y', alpha=0.3)

for i, (tiempo, nombre) in enumerate(zip(tiempos, modelos_nombres)):
    axes[1, 1].text(i, tiempo + max(tiempos)*0.02, f'{tiempo:.1f}', 
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(DIRS['visualizaciones'] / 'comparacion_modelos_completa.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"   Guardada: comparacion_modelos_completa.png")

# -----------------------------------------------------------------------------
# Guardar comparación final
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("GUARDANDO COMPARACIÓN FINAL")
print("-"*80)

comparacion_final = {
    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'mejor_modelo_f1': mejor_modelo_global,
    'mejor_f1_score': float(mejor_f1),
    'mejor_modelo_fn': mejor_fn,
    'menor_fn': int(menor_fn),
    'modelos_evaluados': len(metricas_todos),
    'metricas_por_modelo': {}
}

for modelo, metricas in metricas_todos.items():
    comparacion_final['metricas_por_modelo'][modelo] = {
        'f1_score': float(metricas['metricas_test']['f1_score']),
        'precision': float(metricas['metricas_test']['precision']),
        'recall': float(metricas['metricas_test']['recall']),
        'fn': int(metricas['matriz_confusion']['FN']),
        'tp': int(metricas['matriz_confusion']['TP']),
        'tiempo_minutos': float(metricas.get('tiempo_minutos', 0.0))
    }

with open(DIRS['resultados'] / 'comparacion_final_modelos.json', 'w') as f:
    json.dump(comparacion_final, f, indent=4)

print(f"\nArchivo guardado: comparacion_final_modelos.json")

# Crear tabla CSV
tabla_comparativa = pd.DataFrame({
    'Modelo': list(metricas_todos.keys()),
    'F1_Score': [metricas_todos[m]['metricas_test']['f1_score'] for m in metricas_todos],
    'Precision': [metricas_todos[m]['metricas_test']['precision'] for m in metricas_todos],
    'Recall': [metricas_todos[m]['metricas_test']['recall'] for m in metricas_todos],
    'Accuracy': [metricas_todos[m]['metricas_test']['accuracy'] for m in metricas_todos],
    'ROC_AUC': [metricas_todos[m]['metricas_test']['roc_auc'] for m in metricas_todos],
    'FN': [metricas_todos[m]['matriz_confusion']['FN'] for m in metricas_todos],
    'TP': [metricas_todos[m]['matriz_confusion']['TP'] for m in metricas_todos],
    'Tiempo_min': [metricas_todos[m].get('tiempo_minutos', 0.0) for m in metricas_todos]
})

tabla_comparativa.to_csv(DIRS['resultados'] / 'tabla_comparativa_modelos.csv', index=False)

print(f"Archivo guardado: tabla_comparativa_modelos.csv")

# -----------------------------------------------------------------------------
# Resumen
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("RESUMEN EJECUTIVO")
print("="*80)

print(f"""
PREDICCIÓN DE DESERCIÓN ESTUDIANTIL
Universidad Central del Ecuador

MODELOS EVALUADOS: {len(metricas_todos)}
   - Decision Tree
   - Random Forest
   - Gradient Boosting
   - Regresión Logística
   - XGBoost

MODELO GANADOR: {mejor_modelo_global}
   F1-Score: {mejor_f1:.4f}
   Falsos Negativos: {fn_counts[mejor_modelo_global]:,}

MEJOR EN DETECCIÓN (Menor FN): {mejor_fn}
   FN: {menor_fn:,} desertores no detectados

ARCHIVOS GENERADOS:
   - comparacion_final_modelos.json
   - tabla_comparativa_modelos.csv
   - comparacion_modelos_completa.png

PROYECTO COMPLETADO EXITOSAMENTE
Todos los modelos entrenados y comparados
""")

print("="*80)
print("ANÁLISIS COMPLETO FINALIZADO")
print("="*80)
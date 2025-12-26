"""
ÁRBOLES DE DECISIÓN - ANÁLISIS COMPLETO
Predicción de deserción con árbol de decisión y validación cruzada
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import sys
import time
import pickle
from datetime import datetime

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, roc_curve)
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# Importar configuración
sys.path.insert(0, str(Path(__file__).parent))
exec(open(Path(__file__).parent / '00_config.py').read())

print("="*80)
print("Árboles de decisiÓn")
print("Predicción de deserción con árbol de decisión")
print("="*80)

# -----------------------------------------------------------------------------
# Cargar datos y configuración
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("CARGANDO DATOS Y CONFIGURACIÓN")
print("-"*80)

# Cargar splits
with open(DIRS['datos'] / 'train_test_splits.pkl', 'rb') as f:
    splits = pickle.load(f)

X_train = splits['X_train']
X_test = splits['X_test']
y_train = splits['y_train']
y_test = splits['y_test']
sample_weights_train = splits['sample_weights_train']
sample_weights_test = splits['sample_weights_test']

# Cargar preprocessor
with open(DIRS['modelos'] / 'preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Cargar config features
with open(DIRS['resultados'] / 'config_features.pkl', 'rb') as f:
    config = pickle.load(f)

cat_cols = config['cat_cols']
num_cols = config['num_cols']

print(f"\nDatos cargados:")
print(f"   X_train: {X_train.shape}")
print(f"   X_test:  {X_test.shape}")
print(f"   y_train: {y_train.shape} - Desertores: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
print(f"   y_test:  {y_test.shape} - Desertores: {y_test.sum():,} ({y_test.mean()*100:.2f}%)")

print(f"\nValidación cruzada:")
print(f"   Estrategia: StratifiedKFold")
print(f"   Folds: {N_FOLDS}")
print(f"   Random state: {RANDOM_STATE}")

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# -----------------------------------------------------------------------------
# Crear pipeline
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("PIPELINE DE PREPROCESAMIENTO")
print("-"*80)

pipeline_dt = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', DecisionTreeClassifier(random_state=RANDOM_STATE))
])

print(f"\nPipeline creado:")
print(f"   1. Preprocessor")
print(f"   2. DecisionTreeClassifier")

# -----------------------------------------------------------------------------
# Grid Search
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("GRID SEARCH DE HIPERPARÁMETROS")
print("-"*80)

param_grid_dt = {
    'clf__max_depth': [5, 10, 15, 20, None],
    'clf__min_samples_split': [50, 100, 200],
    'clf__min_samples_leaf': [20, 50, 100],
    'clf__criterion': ['gini', 'entropy']
}

n_combinaciones = 5 * 3 * 3 * 2
print(f"\nParámetros a evaluar:")
print(f"   max_depth: {param_grid_dt['clf__max_depth']}")
print(f"   min_samples_split: {param_grid_dt['clf__min_samples_split']}")
print(f"   min_samples_leaf: {param_grid_dt['clf__min_samples_leaf']}")
print(f"   criterion: {param_grid_dt['clf__criterion']}")

print(f"\nConfiguración del Grid Search:")
print(f"   Total combinaciones: {n_combinaciones}")
print(f"   Entrenamientos totales: {n_combinaciones * N_FOLDS}")
print(f"   Métrica: F1-Score")

print(f"\nIniciando Grid Search...")
inicio_dt = time.time()

grid_dt = GridSearchCV(
    pipeline_dt,
    param_grid_dt,
    cv=cv,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_dt.fit(X_train, y_train, clf__sample_weight=sample_weights_train)

tiempo_dt = time.time() - inicio_dt

print(f"\nGrid Search completado en {tiempo_dt/60:.2f} minutos")

print(f"\nMejores parámetros:")
for param, valor in grid_dt.best_params_.items():
    print(f"   {param}: {valor}")

print(f"\nMejor F1-Score (CV): {grid_dt.best_score_:.4f}")

# -----------------------------------------------------------------------------
# Predicciones
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("GENERANDO PREDICCIONES")
print("-"*80)

print(f"\nPrediciendo en train...")
y_pred_dt_train = grid_dt.predict(X_train)
y_proba_dt_train = grid_dt.predict_proba(X_train)[:, 1]

print(f"Prediciendo en test...")
y_pred_dt_test = grid_dt.predict(X_test)
y_proba_dt_test = grid_dt.predict_proba(X_test)[:, 1]

# -----------------------------------------------------------------------------
# Métricas
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("EVALUACIÓN DE MÉTRICAS")
print("-"*80)

# Train
print(f"\nMÉTRICAS EN TRAIN:")
acc_dt_train = accuracy_score(y_train, y_pred_dt_train)
prec_dt_train = precision_score(y_train, y_pred_dt_train)
rec_dt_train = recall_score(y_train, y_pred_dt_train)
f1_dt_train = f1_score(y_train, y_pred_dt_train)
roc_dt_train = roc_auc_score(y_train, y_proba_dt_train)

print(f"   Accuracy:  {acc_dt_train:.4f}")
print(f"   Precision: {prec_dt_train:.4f}")
print(f"   Recall:    {rec_dt_train:.4f}")
print(f"   F1-Score:  {f1_dt_train:.4f}")
print(f"   ROC-AUC:   {roc_dt_train:.4f}")

# Test
print(f"\nMÉTRICAS EN TEST:")
acc_dt = accuracy_score(y_test, y_pred_dt_test)
prec_dt = precision_score(y_test, y_pred_dt_test)
rec_dt = recall_score(y_test, y_pred_dt_test)
f1_dt = f1_score(y_test, y_pred_dt_test)
roc_dt = roc_auc_score(y_test, y_proba_dt_test)

print(f"   Accuracy:  {acc_dt:.4f}")
print(f"   Precision: {prec_dt:.4f}")
print(f"   Recall:    {rec_dt:.4f}")
print(f"   F1-Score:  {f1_dt:.4f}")
print(f"   ROC-AUC:   {roc_dt:.4f}")

# Overfitting
print(f"\nANÁLISIS DE OVERFITTING:")
diff_f1 = abs(f1_dt_train - f1_dt)
print(f"   Diferencia F1: {diff_f1:.4f}")

if diff_f1 < 0.05:
    estado_overfitting = "EXCELENTE - Sin overfitting"
elif diff_f1 < 0.10:
    estado_overfitting = "BUENO - Overfitting mínimo"
else:
    estado_overfitting = "ADVERTENCIA - Revisar overfitting"

print(f"   Estado: {estado_overfitting}")

# Matriz de confusión
cm_dt = confusion_matrix(y_test, y_pred_dt_test)
tn_dt, fp_dt, fn_dt, tp_dt = cm_dt.ravel()

print(f"\nMATRIZ DE CONFUSIÓN (TEST):")
print(f"\n                    Predicción")
print(f"                 No Desertor  Desertor")
print(f"Real No Desertor    {tn_dt:6,}      {fp_dt:6,}")
print(f"     Desertor       {fn_dt:6,}      {tp_dt:6,}")

print(f"\nINTERPRETACIÓN:")
print(f"   TN: {tn_dt:,} - No desertores correctamente identificados")
print(f"   FP: {fp_dt:,} - Falsa alarma")
print(f"   FN: {fn_dt:,} - Desertores NO detectados (CRÍTICO)")
print(f"   TP: {tp_dt:,} - Desertores detectados correctamente")

especificidad_dt = tn_dt / (tn_dt + fp_dt)
vpp_dt = tp_dt / (tp_dt + fp_dt) if (tp_dt + fp_dt) > 0 else 0
vpn_dt = tn_dt / (tn_dt + fn_dt) if (tn_dt + fn_dt) > 0 else 0

print(f"\nMÉTRICAS ADICIONALES:")
print(f"   Especificidad: {especificidad_dt:.4f}")
print(f"   VPP: {vpp_dt:.4f}")
print(f"   VPN: {vpn_dt:.4f}")

# -----------------------------------------------------------------------------
# Importancia de features
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("IMPORTANCIA DE FEATURES")
print("-"*80)

print(f"\nExtrayendo importancia...")

best_model = grid_dt.best_estimator_
decision_tree = best_model.named_steps['clf']

# Nombres de features
feature_names = []
cat_encoder = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
cat_feature_names = cat_encoder.get_feature_names_out(cat_cols)
feature_names.extend(cat_feature_names)
feature_names.extend(num_cols)

# Importancias
importances = decision_tree.feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print(f"\nTop 20 FACTORES MÁS DETERMINANTES:")
print(f"\n{'Rank':>5s} {'Feature':50s} {'Importance':>12s}")
print("-" * 70)

for idx, row in importance_df.head(20).iterrows():
    print(f"{importance_df.index.get_loc(idx)+1:>5d} {row['Feature']:50s} {row['Importance']:>12.6f}")

# -----------------------------------------------------------------------------
# Visualizaciones
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("GENERANDO VISUALIZACIONES")
print("-"*80)

print("\nVisualizando métricas...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Decision Tree - Análisis Completo', fontsize=16, fontweight='bold')

# Gráfico 1: Train vs Test
metricas = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
train_vals = [acc_dt_train, prec_dt_train, rec_dt_train, f1_dt_train, roc_dt_train]
test_vals = [acc_dt, prec_dt, rec_dt, f1_dt, roc_dt]

x = np.arange(len(metricas))
width = 0.35

axes[0, 0].bar(x - width/2, train_vals, width, label='Train', color='steelblue', edgecolor='black', alpha=0.8)
axes[0, 0].bar(x + width/2, test_vals, width, label='Test', color='coral', edgecolor='black', alpha=0.8)

axes[0, 0].set_ylabel('Score', fontweight='bold', fontsize=12)
axes[0, 0].set_title('Comparación Train vs Test', fontweight='bold', fontsize=13)
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(metricas, rotation=45, ha='right')
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)
axes[0, 0].set_ylim([0, 1.1])

# Gráfico 2: Matriz de confusión
im = axes[0, 1].imshow(cm_dt, cmap='Blues', aspect='auto')
axes[0, 1].set_xticks([0, 1])
axes[0, 1].set_yticks([0, 1])
axes[0, 1].set_xticklabels(['No Desertor', 'Desertor'])
axes[0, 1].set_yticklabels(['No Desertor', 'Desertor'])
axes[0, 1].set_xlabel('Predicción', fontweight='bold', fontsize=12)
axes[0, 1].set_ylabel('Real', fontweight='bold', fontsize=12)
axes[0, 1].set_title('Matriz de Confusión', fontweight='bold', fontsize=13)

for i in range(2):
    for j in range(2):
        axes[0, 1].text(j, i, f'{cm_dt[i, j]:,}',
                       ha="center", va="center", color="black", fontsize=14, fontweight='bold')

plt.colorbar(im, ax=axes[0, 1])

# Gráfico 3: Top 15 features
top_15 = importance_df.head(15)
axes[1, 0].barh(range(len(top_15)), top_15['Importance'].values, color='steelblue', edgecolor='black', alpha=0.7)
axes[1, 0].set_yticks(range(len(top_15)))
axes[1, 0].set_yticklabels(top_15['Feature'].values, fontsize=9)
axes[1, 0].set_xlabel('Importance', fontweight='bold', fontsize=12)
axes[1, 0].set_title('Top 15 Features', fontweight='bold', fontsize=13)
axes[1, 0].invert_yaxis()
axes[1, 0].grid(axis='x', alpha=0.3)

# Gráfico 4: ROC
fpr, tpr, _ = roc_curve(y_test, y_proba_dt_test)
axes[1, 1].plot(fpr, tpr, linewidth=3, color='#e74c3c', label=f'ROC (AUC = {roc_dt:.4f})')
axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[1, 1].set_xlabel('Tasa Falsos Positivos', fontweight='bold', fontsize=12)
axes[1, 1].set_ylabel('Tasa Verdaderos Positivos', fontweight='bold', fontsize=12)
axes[1, 1].set_title('Curva ROC', fontweight='bold', fontsize=13)
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(DIRS['visualizaciones'] / 'decision_tree_completo.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"   Guardada: decision_tree_completo.png")

# -----------------------------------------------------------------------------
#Guardar resultados
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("GUARDANDO RESULTADOS")
print("-"*80)

# Guardar modelo
with open(DIRS['modelos'] / 'decision_tree.pkl', 'wb') as f:
    pickle.dump(grid_dt, f)

# Guardar importancia
importance_df.to_csv(DIRS['resultados'] / 'feature_importance_decision_tree.csv', index=False)

# Guardar métricas
metricas_dt = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'modelo': 'Decision Tree',
    'hiperparametros': {
        'mejor_configuracion': grid_dt.best_params_,
        'n_combinaciones': n_combinaciones
    },
    'metricas_train': {
        'accuracy': float(acc_dt_train),
        'precision': float(prec_dt_train),
        'recall': float(rec_dt_train),
        'f1_score': float(f1_dt_train),
        'roc_auc': float(roc_dt_train)
    },
    'metricas_test': {
        'accuracy': float(acc_dt),
        'precision': float(prec_dt),
        'recall': float(rec_dt),
        'f1_score': float(f1_dt),
        'roc_auc': float(roc_dt)
    },
    'matriz_confusion': {
        'TN': int(tn_dt),
        'FP': int(fp_dt),
        'FN': int(fn_dt),
        'TP': int(tp_dt)
    },
    'overfitting': {
        'diferencia_f1': float(diff_f1),
        'estado': estado_overfitting
    },
    'cv_score': float(grid_dt.best_score_),
    'tiempo_minutos': float(tiempo_dt/60)
}

guardar_resultados_json(metricas_dt, 'metricas_decision_tree')

print(f"\nArchivos guardados:")
print(f"   - decision_tree.pkl")
print(f"   - metricas_decision_tree.json")
print(f"   - feature_importance_decision_tree.csv")

# -----------------------------------------------------------------------------
# Resumen
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("RESUMEN - DECISION TREE")
print("="*80)

print(f"""
MODELO ENTRENADO Y EVALUADO

CONFIGURACIÓN:
   Grid Search: {n_combinaciones} combinaciones
   Validación cruzada: {N_FOLDS}-fold
   Tiempo: {tiempo_dt/60:.2f} minutos

MEJOR CONFIGURACIÓN:
   {grid_dt.best_params_}

RESULTADOS (TEST):
   F1-Score:  {f1_dt:.4f}
   Precision: {prec_dt:.4f}
   Recall:    {rec_dt:.4f}
   ROC-AUC:   {roc_dt:.4f}

MATRIZ DE CONFUSIÓN:
   TP: {tp_dt:,} | FN: {fn_dt:,}
   
TOP 3 FACTORES:
   1. {importance_df.iloc[0]['Feature']}
   2. {importance_df.iloc[1]['Feature']}
   3. {importance_df.iloc[2]['Feature']}

OVERFITTING: {estado_overfitting}

PRÓXIMO PASO:
   Ejecutar src/05_random_forest.py (BLOQUE 5B)
""")

print("="*80)
print("COMPLETADO EXITOSAMENTE")
print("="*80)
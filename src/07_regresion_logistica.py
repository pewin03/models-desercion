"""
REGRESIÓN LOGÍSTICA
Modelo baseline para comparación con modelos de árbol
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

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, roc_curve)
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# Importar configuración
sys.path.insert(0, str(Path(__file__).parent))
exec(open(Path(__file__).parent / '00_config.py').read())

print("="*80)
print("REGRESIÓN LOGÍSTICA")
print("Modelo baseline lineal para comparación")
print("="*80)

# -----------------------------------------------------------------------------
# Cargar datos
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("CARGANDO DATOS")
print("-"*80)

with open(DIRS['datos'] / 'train_test_splits.pkl', 'rb') as f:
    splits = pickle.load(f)

X_train = splits['X_train']
X_test = splits['X_test']
y_train = splits['y_train']
y_test = splits['y_test']
sample_weights_train = splits['sample_weights_train']

with open(DIRS['modelos'] / 'preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open(DIRS['resultados'] / 'config_features.pkl', 'rb') as f:
    config = pickle.load(f)

cat_cols = config['cat_cols']
num_cols = config['num_cols']

print(f"\nDatos cargados:")
print(f"   X_train: {X_train.shape}")
print(f"   X_test:  {X_test.shape}")

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("PIPELINE")
print("-"*80)

pipeline_lr = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, n_jobs=-1))
])

print(f"\nPipeline creado con LogisticRegression")

# -----------------------------------------------------------------------------
# Grid Search
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("GRID SEARCH")
print("-"*80)

param_grid_lr = {
    'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'clf__penalty': ['l2'],
    'clf__solver': ['lbfgs', 'liblinear']
}

n_combinaciones_lr = 6 * 1 * 2

print(f"\nParámetros:")
print(f"   C: {param_grid_lr['clf__C']}")
print(f"   penalty: {param_grid_lr['clf__penalty']}")
print(f"   solver: {param_grid_lr['clf__solver']}")

print(f"\nConfiguración:")
print(f"   Total combinaciones: {n_combinaciones_lr}")
print(f"   Tiempo estimado: 2-5 minutos")

print(f"\nIniciando Grid Search...")
inicio_lr = time.time()

grid_lr = GridSearchCV(
    pipeline_lr,
    param_grid_lr,
    cv=cv,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_lr.fit(X_train, y_train, clf__sample_weight=sample_weights_train)

tiempo_lr = time.time() - inicio_lr

print(f"\nGrid Search completado en {tiempo_lr/60:.2f} minutos")

print(f"\nMejores parámetros:")
for param, valor in grid_lr.best_params_.items():
    print(f"   {param}: {valor}")

print(f"\nMejor F1-Score (CV): {grid_lr.best_score_:.4f}")

# -----------------------------------------------------------------------------
# Predicciones
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("PREDICCIONES")
print("-"*80)

y_pred_lr_train = grid_lr.predict(X_train)
y_proba_lr_train = grid_lr.predict_proba(X_train)[:, 1]

y_pred_lr_test = grid_lr.predict(X_test)
y_proba_lr_test = grid_lr.predict_proba(X_test)[:, 1]

# -----------------------------------------------------------------------------
# Paso 5: Métricas
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("MÉTRICAS")
print("-"*80)

# Train
print(f"\nTRAIN:")
acc_lr_train = accuracy_score(y_train, y_pred_lr_train)
prec_lr_train = precision_score(y_train, y_pred_lr_train)
rec_lr_train = recall_score(y_train, y_pred_lr_train)
f1_lr_train = f1_score(y_train, y_pred_lr_train)
roc_lr_train = roc_auc_score(y_train, y_proba_lr_train)

print(f"   Accuracy:  {acc_lr_train:.4f}")
print(f"   Precision: {prec_lr_train:.4f}")
print(f"   Recall:    {rec_lr_train:.4f}")
print(f"   F1-Score:  {f1_lr_train:.4f}")
print(f"   ROC-AUC:   {roc_lr_train:.4f}")

# Test
print(f"\nTEST:")
acc_lr = accuracy_score(y_test, y_pred_lr_test)
prec_lr = precision_score(y_test, y_pred_lr_test)
rec_lr = recall_score(y_test, y_pred_lr_test)
f1_lr = f1_score(y_test, y_pred_lr_test)
roc_lr = roc_auc_score(y_test, y_proba_lr_test)

print(f"   Accuracy:  {acc_lr:.4f}")
print(f"   Precision: {prec_lr:.4f}")
print(f"   Recall:    {rec_lr:.4f}")
print(f"   F1-Score:  {f1_lr:.4f}")
print(f"   ROC-AUC:   {roc_lr:.4f}")

# Overfitting
diff_f1_lr = abs(f1_lr_train - f1_lr)
print(f"\nOVERFITTING:")
print(f"   Diferencia F1: {diff_f1_lr:.4f}")

# Matriz
cm_lr = confusion_matrix(y_test, y_pred_lr_test)
tn_lr, fp_lr, fn_lr, tp_lr = cm_lr.ravel()

print(f"\nMATRIZ DE CONFUSIÓN:")
print(f"   TN: {tn_lr:,} | FP: {fp_lr:,}")
print(f"   FN: {fn_lr:,} | TP: {tp_lr:,}")

# -----------------------------------------------------------------------------
# Visualización
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("VISUALIZACIÓN")
print("-"*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Regresión Logística - Análisis Completo', fontsize=16, fontweight='bold')

# Train vs Test
metricas = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
train_vals = [acc_lr_train, prec_lr_train, rec_lr_train, f1_lr_train, roc_lr_train]
test_vals = [acc_lr, prec_lr, rec_lr, f1_lr, roc_lr]

x = np.arange(len(metricas))
width = 0.35

axes[0, 0].bar(x - width/2, train_vals, width, label='Train', color='steelblue', edgecolor='black', alpha=0.8)
axes[0, 0].bar(x + width/2, test_vals, width, label='Test', color='coral', edgecolor='black', alpha=0.8)
axes[0, 0].set_ylabel('Score', fontweight='bold')
axes[0, 0].set_title('Train vs Test', fontweight='bold')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(metricas, rotation=45, ha='right')
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)
axes[0, 0].set_ylim([0, 1.1])

# Matriz
im = axes[0, 1].imshow(cm_lr, cmap='Blues', aspect='auto')
axes[0, 1].set_xticks([0, 1])
axes[0, 1].set_yticks([0, 1])
axes[0, 1].set_xticklabels(['No Desertor', 'Desertor'])
axes[0, 1].set_yticklabels(['No Desertor', 'Desertor'])
axes[0, 1].set_title('Matriz de Confusión', fontweight='bold')

for i in range(2):
    for j in range(2):
        axes[0, 1].text(j, i, f'{cm_lr[i, j]:,}', ha="center", va="center", fontsize=14, fontweight='bold')

plt.colorbar(im, ax=axes[0, 1])

# ROC
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr_test)
axes[1, 0].plot(fpr_lr, tpr_lr, linewidth=3, label=f'ROC (AUC = {roc_lr:.4f})')
axes[1, 0].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[1, 0].set_xlabel('FPR', fontweight='bold')
axes[1, 0].set_ylabel('TPR', fontweight='bold')
axes[1, 0].set_title('Curva ROC', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Coeficientes (top 20)
best_model = grid_lr.best_estimator_
logistic = best_model.named_steps['clf']

feature_names_lr = []
cat_encoder = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
cat_feature_names = cat_encoder.get_feature_names_out(cat_cols)
feature_names_lr.extend(cat_feature_names)
feature_names_lr.extend(num_cols)

coefs = logistic.coef_[0]
coef_df = pd.DataFrame({
    'Feature': feature_names_lr,
    'Coefficient': np.abs(coefs)
}).sort_values('Coefficient', ascending=False)

top_20 = coef_df.head(20)
axes[1, 1].barh(range(len(top_20)), top_20['Coefficient'].values, color='steelblue', edgecolor='black', alpha=0.7)
axes[1, 1].set_yticks(range(len(top_20)))
axes[1, 1].set_yticklabels(top_20['Feature'].values, fontsize=9)
axes[1, 1].set_xlabel('|Coefficient|', fontweight='bold')
axes[1, 1].set_title('Top 20 Coeficientes', fontweight='bold')
axes[1, 1].invert_yaxis()
axes[1, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(DIRS['visualizaciones'] / 'regresion_logistica_completo.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"   Guardada: regresion_logistica_completo.png")

# -----------------------------------------------------------------------------
# Guardar
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("GUARDANDO")
print("-"*80)

with open(DIRS['modelos'] / 'regresion_logistica.pkl', 'wb') as f:
    pickle.dump(grid_lr, f)

metricas_lr = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'modelo': 'Regresion Logistica',
    'hiperparametros': {
        'mejor_configuracion': grid_lr.best_params_,
        'n_combinaciones': n_combinaciones_lr
    },
    'metricas_test': {
        'accuracy': float(acc_lr),
        'precision': float(prec_lr),
        'recall': float(rec_lr),
        'f1_score': float(f1_lr),
        'roc_auc': float(roc_lr)
    },
    'matriz_confusion': {
        'TN': int(tn_lr),
        'FP': int(fp_lr),
        'FN': int(fn_lr),
        'TP': int(tp_lr)
    },
    'cv_score': float(grid_lr.best_score_),
    'tiempo_minutos': float(tiempo_lr/60)
}

guardar_resultados_json(metricas_lr, 'metricas_regresion_logistica')

print(f"\nArchivos guardados")

# -----------------------------------------------------------------------------
# Resumen
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("RESUMEN - REGRESIÓN LOGÍSTICA")
print("="*80)

print(f"""
MODELO BASELINE COMPLETADO

RESULTADOS (TEST):
   F1-Score:  {f1_lr:.4f}
   Precision: {prec_lr:.4f}
   Recall:    {rec_lr:.4f}

MATRIZ:
   TP: {tp_lr:,} | FN: {fn_lr:,}

TIEMPO: {tiempo_lr/60:.2f} minutos

PRÓXIMO PASO:
   Ejecutar src/08_xgboost.py
""")

print("="*80)
print("COMPLETADO")
print("="*80)
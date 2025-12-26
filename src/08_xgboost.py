"""
XGBOOST
Gradient Boosting optimizado con XGBoost
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import sys
import time
import pickle
from datetime import datetime

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, roc_curve)
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# Importar configuración
sys.path.insert(0, str(Path(__file__).parent))
exec(open(Path(__file__).parent / '00_config.py').read())

print("="*80)
print("XGBOOST")
print("Gradient Boosting optimizado")
print("="*80)

# -----------------------------------------------------------------------------
# Paso 1: Cargar datos
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

pipeline_xgb = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, tree_method='hist'))
])

print(f"\nPipeline creado con XGBClassifier")

# -----------------------------------------------------------------------------
# Grid Search
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("GRID SEARCH")
print("-"*80)

param_grid_xgb = {
    'clf__n_estimators': [100, 200, 300],
    'clf__learning_rate': [0.01, 0.05, 0.1],
    'clf__max_depth': [3, 5, 7],
    'clf__subsample': [0.8, 1.0],
    'clf__colsample_bytree': [0.8, 1.0],
    'clf__min_child_weight': [1, 3, 5]
}

n_combinaciones_xgb = 3 * 3 * 3 * 2 * 2 * 3

print(f"\nParámetros:")
print(f"   n_estimators: {param_grid_xgb['clf__n_estimators']}")
print(f"   learning_rate: {param_grid_xgb['clf__learning_rate']}")
print(f"   max_depth: {param_grid_xgb['clf__max_depth']}")
print(f"   subsample: {param_grid_xgb['clf__subsample']}")
print(f"   colsample_bytree: {param_grid_xgb['clf__colsample_bytree']}")
print(f"   min_child_weight: {param_grid_xgb['clf__min_child_weight']}")

print(f"\nConfiguración:")
print(f"   Total combinaciones: {n_combinaciones_xgb}")
print(f"   Tiempo estimado: 15-30 minutos")

print(f"\nIniciando Grid Search...")
inicio_xgb = time.time()

grid_xgb = GridSearchCV(
    pipeline_xgb,
    param_grid_xgb,
    cv=cv,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

grid_xgb.fit(X_train, y_train, clf__sample_weight=sample_weights_train)

tiempo_xgb = time.time() - inicio_xgb

print(f"\nGrid Search completado en {tiempo_xgb/60:.2f} minutos")

print(f"\nMejores parámetros:")
for param, valor in grid_xgb.best_params_.items():
    print(f"   {param}: {valor}")

print(f"\nMejor F1-Score (CV): {grid_xgb.best_score_:.4f}")

# -----------------------------------------------------------------------------
# Predicciones
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("PREDICCIONES")
print("-"*80)

y_pred_xgb_train = grid_xgb.predict(X_train)
y_proba_xgb_train = grid_xgb.predict_proba(X_train)[:, 1]

y_pred_xgb_test = grid_xgb.predict(X_test)
y_proba_xgb_test = grid_xgb.predict_proba(X_test)[:, 1]

# -----------------------------------------------------------------------------
# Métricas
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("MÉTRICAS")
print("-"*80)

# Train
print(f"\nTRAIN:")
acc_xgb_train = accuracy_score(y_train, y_pred_xgb_train)
prec_xgb_train = precision_score(y_train, y_pred_xgb_train)
rec_xgb_train = recall_score(y_train, y_pred_xgb_train)
f1_xgb_train = f1_score(y_train, y_pred_xgb_train)
roc_xgb_train = roc_auc_score(y_train, y_proba_xgb_train)

print(f"   Accuracy:  {acc_xgb_train:.4f}")
print(f"   Precision: {prec_xgb_train:.4f}")
print(f"   Recall:    {rec_xgb_train:.4f}")
print(f"   F1-Score:  {f1_xgb_train:.4f}")
print(f"   ROC-AUC:   {roc_xgb_train:.4f}")

# Test
print(f"\nTEST:")
acc_xgb = accuracy_score(y_test, y_pred_xgb_test)
prec_xgb = precision_score(y_test, y_pred_xgb_test)
rec_xgb = recall_score(y_test, y_pred_xgb_test)
f1_xgb = f1_score(y_test, y_pred_xgb_test)
roc_xgb = roc_auc_score(y_test, y_proba_xgb_test)

print(f"   Accuracy:  {acc_xgb:.4f}")
print(f"   Precision: {prec_xgb:.4f}")
print(f"   Recall:    {rec_xgb:.4f}")
print(f"   F1-Score:  {f1_xgb:.4f}")
print(f"   ROC-AUC:   {roc_xgb:.4f}")

# Overfitting
diff_f1_xgb = abs(f1_xgb_train - f1_xgb)
print(f"\nOVERFITTING:")
print(f"   Diferencia F1: {diff_f1_xgb:.4f}")

# Matriz
cm_xgb = confusion_matrix(y_test, y_pred_xgb_test)
tn_xgb, fp_xgb, fn_xgb, tp_xgb = cm_xgb.ravel()

print(f"\nMATRIZ DE CONFUSIÓN:")
print(f"   TN: {tn_xgb:,} | FP: {fp_xgb:,}")
print(f"   FN: {fn_xgb:,} | TP: {tp_xgb:,}")

# -----------------------------------------------------------------------------
# Importancia
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("IMPORTANCIA")
print("-"*80)

best_model_xgb = grid_xgb.best_estimator_
xgboost = best_model_xgb.named_steps['clf']

feature_names_xgb = []
cat_encoder_xgb = best_model_xgb.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
cat_feature_names_xgb = cat_encoder_xgb.get_feature_names_out(cat_cols)
feature_names_xgb.extend(cat_feature_names_xgb)
feature_names_xgb.extend(num_cols)

importances_xgb = xgboost.feature_importances_

importance_xgb_df = pd.DataFrame({
    'Feature': feature_names_xgb,
    'Importance': importances_xgb
}).sort_values('Importance', ascending=False)

print(f"\nTop 20:")
for idx, row in importance_xgb_df.head(20).iterrows():
    print(f"   {importance_xgb_df.index.get_loc(idx)+1:2d}. {row['Feature']:50s} {row['Importance']:.6f}")

# -----------------------------------------------------------------------------
# Visualización
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("VISUALIZACIÓN")
print("-"*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('XGBoost - Análisis Completo', fontsize=16, fontweight='bold')

# Train vs Test
metricas = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
train_vals = [acc_xgb_train, prec_xgb_train, rec_xgb_train, f1_xgb_train, roc_xgb_train]
test_vals = [acc_xgb, prec_xgb, rec_xgb, f1_xgb, roc_xgb]

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
im = axes[0, 1].imshow(cm_xgb, cmap='Purples', aspect='auto')
axes[0, 1].set_xticks([0, 1])
axes[0, 1].set_yticks([0, 1])
axes[0, 1].set_xticklabels(['No Desertor', 'Desertor'])
axes[0, 1].set_yticklabels(['No Desertor', 'Desertor'])
axes[0, 1].set_title('Matriz de Confusión', fontweight='bold')

for i in range(2):
    for j in range(2):
        axes[0, 1].text(j, i, f'{cm_xgb[i, j]:,}', ha="center", va="center", fontsize=14, fontweight='bold')

plt.colorbar(im, ax=axes[0, 1])

# Features
top_15 = importance_xgb_df.head(15)
axes[1, 0].barh(range(len(top_15)), top_15['Importance'].values, color='purple', edgecolor='black', alpha=0.7)
axes[1, 0].set_yticks(range(len(top_15)))
axes[1, 0].set_yticklabels(top_15['Feature'].values, fontsize=9)
axes[1, 0].set_xlabel('Importance', fontweight='bold')
axes[1, 0].set_title('Top 15 Features', fontweight='bold')
axes[1, 0].invert_yaxis()
axes[1, 0].grid(axis='x', alpha=0.3)

# ROC
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb_test)
axes[1, 1].plot(fpr_xgb, tpr_xgb, linewidth=3, color='purple', label=f'ROC (AUC = {roc_xgb:.4f})')
axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[1, 1].set_xlabel('FPR', fontweight='bold')
axes[1, 1].set_ylabel('TPR', fontweight='bold')
axes[1, 1].set_title('Curva ROC', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(DIRS['visualizaciones'] / 'xgboost_completo.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"   Guardada: xgboost_completo.png")

# -----------------------------------------------------------------------------
# Guardar
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("GUARDANDO")
print("-"*80)

with open(DIRS['modelos'] / 'xgboost.pkl', 'wb') as f:
    pickle.dump(grid_xgb, f)

importance_xgb_df.to_csv(DIRS['resultados'] / 'feature_importance_xgboost.csv', index=False)

metricas_xgb = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'modelo': 'XGBoost',
    'hiperparametros': {
        'mejor_configuracion': grid_xgb.best_params_,
        'n_combinaciones': n_combinaciones_xgb
    },
    'metricas_test': {
        'accuracy': float(acc_xgb),
        'precision': float(prec_xgb),
        'recall': float(rec_xgb),
        'f1_score': float(f1_xgb),
        'roc_auc': float(roc_xgb)
    },
    'matriz_confusion': {
        'TN': int(tn_xgb),
        'FP': int(fp_xgb),
        'FN': int(fn_xgb),
        'TP': int(tp_xgb)
    },
    'cv_score': float(grid_xgb.best_score_),
    'tiempo_minutos': float(tiempo_xgb/60)
}

guardar_resultados_json(metricas_xgb, 'metricas_xgboost')

print(f"\nArchivos guardados")

# -----------------------------------------------------------------------------
# Resumen
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("RESUMEN - XGBOOST")
print("="*80)

print(f"""
MODELO COMPLETADO

RESULTADOS (TEST):
   F1-Score:  {f1_xgb:.4f}
   Precision: {prec_xgb:.4f}
   Recall:    {rec_xgb:.4f}

MATRIZ:
   TP: {tp_xgb:,} | FN: {fn_xgb:,}

TIEMPO: {tiempo_xgb/60:.2f} minutos

PRÓXIMO PASO:
   Ejecutar src/09_comparacion.py
""")

print("="*80)
print("COMPLETADO")
print("="*80)
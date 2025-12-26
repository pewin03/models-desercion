"""
GRADIENT BOOSTING - ANÁLISIS EXHAUSTIVO
Predicción de deserción con Gradient Boosting y Grid Search exhaustivo
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

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, roc_curve,
                            average_precision_score)
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# Importar configuración
sys.path.insert(0, str(Path(__file__).parent))
exec(open(Path(__file__).parent / '00_config.py').read())

print("="*80)
print("BLOQUE GRADIENT BOOSTING")
print("Predicción de deserción con Gradient Boosting - BÚSQUEDA EXHAUSTIVA")
print("="*80)

# -----------------------------------------------------------------------------
# Cargar datos
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print(" CARGANDO DATOS Y CONFIGURACIÓN")
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

pipeline_gb = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', GradientBoostingClassifier(random_state=RANDOM_STATE))
])

print(f"\nPipeline creado con GradientBoostingClassifier")

# -----------------------------------------------------------------------------
#Grid Search EXHAUSTIVO
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("GRID SEARCH EXHAUSTIVO")
print("-"*80)

param_grid_gb = {
    'clf__n_estimators': [100, 200, 300],
    'clf__learning_rate': [0.01, 0.05, 0.1],
    'clf__max_depth': [3, 5, 7],
    'clf__subsample': [0.6, 0.8, 1.0],
    'clf__min_samples_split': [50, 100, 150],
    'clf__min_samples_leaf': [20, 50, 100],
    'clf__max_features': ['sqrt', 'log2']
}

n_combinaciones_gb = 3 * 3 * 3 * 3 * 3 * 3 * 2

print(f"\nParámetros (BÚSQUEDA EXHAUSTIVA):")
print(f"   n_estimators: {param_grid_gb['clf__n_estimators']}")
print(f"   learning_rate: {param_grid_gb['clf__learning_rate']}")
print(f"   max_depth: {param_grid_gb['clf__max_depth']}")
print(f"   subsample: {param_grid_gb['clf__subsample']}")
print(f"   min_samples_split: {param_grid_gb['clf__min_samples_split']}")
print(f"   min_samples_leaf: {param_grid_gb['clf__min_samples_leaf']}")
print(f"   max_features: {param_grid_gb['clf__max_features']}")

print(f"\nConfiguración:")
print(f"   Total combinaciones: {n_combinaciones_gb}")
print(f"   Entrenamientos totales: {n_combinaciones_gb * N_FOLDS}")
print(f"   Tiempo estimado: 30-60 minutos")

print(f"\n{'!'*80}")
print("BÚSQUEDA EXHAUSTIVA - MÁXIMA CALIDAD CIENTÍFICA")
print(f"{'!'*80}")

print(f"\nIniciando Grid Search EXHAUSTIVO...")
inicio_gb = time.time()

grid_gb = GridSearchCV(
    pipeline_gb,
    param_grid_gb,
    cv=cv,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

grid_gb.fit(X_train, y_train, clf__sample_weight=sample_weights_train)

tiempo_gb = time.time() - inicio_gb

print(f"\nGrid Search completado en {tiempo_gb/60:.2f} minutos")

print(f"\nMejores parámetros:")
for param, valor in grid_gb.best_params_.items():
    print(f"   {param}: {valor}")

print(f"\nMejor F1-Score (CV): {grid_gb.best_score_:.4f}")

# -----------------------------------------------------------------------------
# Predicciones
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("GENERANDO PREDICCIONES")
print("-"*80)

y_pred_gb_train = grid_gb.predict(X_train)
y_proba_gb_train = grid_gb.predict_proba(X_train)[:, 1]

y_pred_gb_test = grid_gb.predict(X_test)
y_proba_gb_test = grid_gb.predict_proba(X_test)[:, 1]

# -----------------------------------------------------------------------------
# Métricas
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("EVALUACIÓN EXHAUSTIVA DE MÉTRICAS")
print("-"*80)

# Train
print(f"\nMÉTRICAS EN TRAIN:")
acc_gb_train = accuracy_score(y_train, y_pred_gb_train)
prec_gb_train = precision_score(y_train, y_pred_gb_train)
rec_gb_train = recall_score(y_train, y_pred_gb_train)
f1_gb_train = f1_score(y_train, y_pred_gb_train)
roc_gb_train = roc_auc_score(y_train, y_proba_gb_train)

print(f"   Accuracy:  {acc_gb_train:.4f}")
print(f"   Precision: {prec_gb_train:.4f}")
print(f"   Recall:    {rec_gb_train:.4f}")
print(f"   F1-Score:  {f1_gb_train:.4f}")
print(f"   ROC-AUC:   {roc_gb_train:.4f}")

# Test
print(f"\nMÉTRICAS EN TEST:")
acc_gb = accuracy_score(y_test, y_pred_gb_test)
prec_gb = precision_score(y_test, y_pred_gb_test)
rec_gb = recall_score(y_test, y_pred_gb_test)
f1_gb = f1_score(y_test, y_pred_gb_test)
roc_gb = roc_auc_score(y_test, y_proba_gb_test)
avg_precision = average_precision_score(y_test, y_proba_gb_test)

print(f"   Accuracy:  {acc_gb:.4f}")
print(f"   Precision: {prec_gb:.4f}")
print(f"   Recall:    {rec_gb:.4f}")
print(f"   F1-Score:  {f1_gb:.4f}")
print(f"   ROC-AUC:   {roc_gb:.4f}")

# Overfitting
diff_f1_gb = abs(f1_gb_train - f1_gb)
print(f"\nANÁLISIS DE OVERFITTING:")
print(f"   Diferencia F1: {diff_f1_gb:.4f}")

if diff_f1_gb < 0.05:
    estado_overfitting_gb = "EXCELENTE - Sin overfitting"
elif diff_f1_gb < 0.10:
    estado_overfitting_gb = "BUENO - Overfitting mínimo"
else:
    estado_overfitting_gb = "ADVERTENCIA - Revisar overfitting"

print(f"   Estado: {estado_overfitting_gb}")

# Matriz
cm_gb = confusion_matrix(y_test, y_pred_gb_test)
tn_gb, fp_gb, fn_gb, tp_gb = cm_gb.ravel()

print(f"\nMATRIZ DE CONFUSIÓN (TEST):")
print(f"\n                    Predicción")
print(f"                 No Desertor  Desertor")
print(f"Real No Desertor    {tn_gb:6,}      {fp_gb:6,}")
print(f"     Desertor       {fn_gb:6,}      {tp_gb:6,}")

especificidad_gb = tn_gb / (tn_gb + fp_gb)
vpp_gb = tp_gb / (tp_gb + fp_gb) if (tp_gb + fp_gb) > 0 else 0
vpn_gb = tn_gb / (tn_gb + fn_gb) if (tn_gb + fn_gb) > 0 else 0

print(f"\nMÉTRICAS ADICIONALES:")
print(f"   Especificidad: {especificidad_gb:.4f}")
print(f"   VPP: {vpp_gb:.4f}")
print(f"   VPN: {vpn_gb:.4f}")

# -----------------------------------------------------------------------------
# Importancia
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("IMPORTANCIA DE FEATURES")
print("-"*80)

best_model_gb = grid_gb.best_estimator_
gradient_boosting = best_model_gb.named_steps['clf']

feature_names_gb = []
cat_encoder_gb = best_model_gb.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
cat_feature_names_gb = cat_encoder_gb.get_feature_names_out(cat_cols)
feature_names_gb.extend(cat_feature_names_gb)
feature_names_gb.extend(num_cols)

importances_gb = gradient_boosting.feature_importances_

importance_gb_df = pd.DataFrame({
    'Feature': feature_names_gb,
    'Importance': importances_gb
}).sort_values('Importance', ascending=False)

print(f"\nTop 30 FACTORES:")
print(f"\n{'Rank':>5s} {'Feature':50s} {'Importance':>12s}")
print("-" * 70)

for idx, row in importance_gb_df.head(30).iterrows():
    print(f"{importance_gb_df.index.get_loc(idx)+1:>5d} {row['Feature']:50s} {row['Importance']:>12.6f}")

importancia_top5 = importance_gb_df.head(5)['Importance'].sum()
importancia_top10 = importance_gb_df.head(10)['Importance'].sum()
importancia_top20 = importance_gb_df.head(20)['Importance'].sum()

print(f"\nCONCENTRACIÓN:")
print(f"   Top 5:  {importancia_top5*100:.2f}%")
print(f"   Top 10: {importancia_top10*100:.2f}%")
print(f"   Top 20: {importancia_top20*100:.2f}%")

# -----------------------------------------------------------------------------
# Visualización
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("GENERANDO VISUALIZACIÓN")
print("-"*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Gradient Boosting - Análisis Completo', fontsize=16, fontweight='bold')

# Train vs Test
metricas = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
train_vals_gb = [acc_gb_train, prec_gb_train, rec_gb_train, f1_gb_train, roc_gb_train]
test_vals_gb = [acc_gb, prec_gb, rec_gb, f1_gb, roc_gb]

x = np.arange(len(metricas))
width = 0.35

axes[0, 0].bar(x - width/2, train_vals_gb, width, label='Train', color='steelblue', edgecolor='black', alpha=0.8)
axes[0, 0].bar(x + width/2, test_vals_gb, width, label='Test', color='coral', edgecolor='black', alpha=0.8)
axes[0, 0].set_ylabel('Score', fontweight='bold', fontsize=12)
axes[0, 0].set_title('Train vs Test', fontweight='bold', fontsize=13)
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(metricas, rotation=45, ha='right')
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)
axes[0, 0].set_ylim([0, 1.1])

# Matriz
im = axes[0, 1].imshow(cm_gb, cmap='Reds', aspect='auto')
axes[0, 1].set_xticks([0, 1])
axes[0, 1].set_yticks([0, 1])
axes[0, 1].set_xticklabels(['No Desertor', 'Desertor'])
axes[0, 1].set_yticklabels(['No Desertor', 'Desertor'])
axes[0, 1].set_xlabel('Predicción', fontweight='bold', fontsize=12)
axes[0, 1].set_ylabel('Real', fontweight='bold', fontsize=12)
axes[0, 1].set_title('Matriz de Confusión', fontweight='bold', fontsize=13)

for i in range(2):
    for j in range(2):
        axes[0, 1].text(j, i, f'{cm_gb[i, j]:,}',
                       ha="center", va="center", color="black", fontsize=14, fontweight='bold')

plt.colorbar(im, ax=axes[0, 1])

# Features
top_15_gb = importance_gb_df.head(15)
axes[1, 0].barh(range(len(top_15_gb)), top_15_gb['Importance'].values, color='#e74c3c', edgecolor='black', alpha=0.7)
axes[1, 0].set_yticks(range(len(top_15_gb)))
axes[1, 0].set_yticklabels(top_15_gb['Feature'].values, fontsize=9)
axes[1, 0].set_xlabel('Importance', fontweight='bold', fontsize=12)
axes[1, 0].set_title('Top 15 Features', fontweight='bold', fontsize=13)
axes[1, 0].invert_yaxis()
axes[1, 0].grid(axis='x', alpha=0.3)

# ROC
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_proba_gb_test)
axes[1, 1].plot(fpr_gb, tpr_gb, linewidth=3, color='#e74c3c', label=f'ROC (AUC = {roc_gb:.4f})')
axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[1, 1].set_xlabel('Tasa Falsos Positivos', fontweight='bold', fontsize=12)
axes[1, 1].set_ylabel('Tasa Verdaderos Positivos', fontweight='bold', fontsize=12)
axes[1, 1].set_title('Curva ROC', fontweight='bold', fontsize=13)
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(DIRS['visualizaciones'] / 'gradient_boosting_completo.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"   Guardada: gradient_boosting_completo.png")

# -----------------------------------------------------------------------------
# uardar
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("GUARDANDO RESULTADOS")
print("-"*80)

with open(DIRS['modelos'] / 'gradient_boosting.pkl', 'wb') as f:
    pickle.dump(grid_gb, f)

importance_gb_df.to_csv(DIRS['resultados'] / 'feature_importance_gradient_boosting.csv', index=False)

metricas_gb = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'modelo': 'Gradient Boosting',
    'tipo_busqueda': 'exhaustiva',
    'hiperparametros': {
        'mejor_configuracion': grid_gb.best_params_,
        'n_combinaciones': n_combinaciones_gb
    },
    'metricas_test': {
        'accuracy': float(acc_gb),
        'precision': float(prec_gb),
        'recall': float(rec_gb),
        'f1_score': float(f1_gb),
        'roc_auc': float(roc_gb)
    },
    'matriz_confusion': {
        'TN': int(tn_gb),
        'FP': int(fp_gb),
        'FN': int(fn_gb),
        'TP': int(tp_gb)
    },
    'overfitting': {
        'diferencia_f1': float(diff_f1_gb),
        'estado': estado_overfitting_gb
    },
    'cv_score': float(grid_gb.best_score_),
    'tiempo_minutos': float(tiempo_gb/60)
}

guardar_resultados_json(metricas_gb, 'metricas_gradient_boosting')

print(f"\nArchivos guardados:")
print(f"   - gradient_boosting.pkl")
print(f"   - metricas_gradient_boosting.json")
print(f"   - feature_importance_gradient_boosting.csv")

# -----------------------------------------------------------------------------
# Resumen
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("RESUMEN - GRADIENT BOOSTING")
print("="*80)

print(f"""
BÚSQUEDA EXHAUSTIVA COMPLETADA

CONFIGURACIÓN:
   Combinaciones: {n_combinaciones_gb}
   Tiempo: {tiempo_gb/60:.2f} minutos

RESULTADOS (TEST):
   F1-Score:  {f1_gb:.4f}
   Precision: {prec_gb:.4f}
   Recall:    {rec_gb:.4f}
   ROC-AUC:   {roc_gb:.4f}

MATRIZ:
   TP: {tp_gb:,} | FN: {fn_gb:,}

TOP 3:
   1. {importance_gb_df.iloc[0]['Feature']}
   2. {importance_gb_df.iloc[1]['Feature']}
   3. {importance_gb_df.iloc[2]['Feature']}

OVERFITTING: {estado_overfitting_gb}

PRÓXIMO PASO:
   Ejecutar src/07_regresion_logistica.py
""")

print("="*80)
print("COMPLETADO EXITOSAMENTE")
print("="*80)
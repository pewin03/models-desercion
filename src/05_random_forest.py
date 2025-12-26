"""
RANDOM FOREST - ANÁLISIS COMPLETO
Predicción de deserción con ensamble de árboles y validación cruzada
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, roc_curve)
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# Importar configuración
sys.path.insert(0, str(Path(__file__).parent))
exec(open(Path(__file__).parent / '00_config.py').read())

print("="*80)
print("RANDOM FOREST")
print("Predicción de deserción con ensamble de árboles")
print("="*80)

# -----------------------------------------------------------------------------
# Cargar datos
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("CARGANDO DATOS Y CONFIGURACIÓN")
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
print(f"   y_train: Desertores: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
print(f"   y_test:  Desertores: {y_test.sum():,} ({y_test.mean()*100:.2f}%)")

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("PIPELINE")
print("-"*80)

pipeline_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
])

print(f"\nPipeline creado con RandomForestClassifier")

# -----------------------------------------------------------------------------
# Grid Search
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("GRID SEARCH DE HIPERPARÁMETROS")
print("-"*80)

param_grid_rf = {
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [10, 20, 30, None],
    'clf__min_samples_split': [50, 100, 200],
    'clf__min_samples_leaf': [20, 50, 100],
    'clf__max_features': ['sqrt', 'log2']
}

n_combinaciones_rf = 3 * 4 * 3 * 3 * 2

print(f"\nParámetros a evaluar:")
print(f"   n_estimators: {param_grid_rf['clf__n_estimators']}")
print(f"   max_depth: {param_grid_rf['clf__max_depth']}")
print(f"   min_samples_split: {param_grid_rf['clf__min_samples_split']}")
print(f"   min_samples_leaf: {param_grid_rf['clf__min_samples_leaf']}")
print(f"   max_features: {param_grid_rf['clf__max_features']}")

print(f"\nConfiguración:")
print(f"   Total combinaciones: {n_combinaciones_rf}")
print(f"   Entrenamientos totales: {n_combinaciones_rf * N_FOLDS}")
print(f"   Tiempo estimado: 15-25 minutos")

print(f"\nIniciando Grid Search...")
inicio_rf = time.time()

grid_rf = GridSearchCV(
    pipeline_rf,
    param_grid_rf,
    cv=cv,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

grid_rf.fit(X_train, y_train, clf__sample_weight=sample_weights_train)

tiempo_rf = time.time() - inicio_rf

print(f"\nGrid Search completado en {tiempo_rf/60:.2f} minutos")

print(f"\nMejores parámetros:")
for param, valor in grid_rf.best_params_.items():
    print(f"   {param}: {valor}")

print(f"\nMejor F1-Score (CV): {grid_rf.best_score_:.4f}")

# -----------------------------------------------------------------------------
# Predicciones
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("PASO 4: GENERANDO PREDICCIONES")
print("-"*80)

print(f"\nPrediciendo en train...")
y_pred_rf_train = grid_rf.predict(X_train)
y_proba_rf_train = grid_rf.predict_proba(X_train)[:, 1]

print(f"Prediciendo en test...")
y_pred_rf_test = grid_rf.predict(X_test)
y_proba_rf_test = grid_rf.predict_proba(X_test)[:, 1]

# -----------------------------------------------------------------------------
# Métricas
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("EVALUACIÓN DE MÉTRICAS")
print("-"*80)

# Train
print(f"\nMÉTRICAS EN TRAIN:")
acc_rf_train = accuracy_score(y_train, y_pred_rf_train)
prec_rf_train = precision_score(y_train, y_pred_rf_train)
rec_rf_train = recall_score(y_train, y_pred_rf_train)
f1_rf_train = f1_score(y_train, y_pred_rf_train)
roc_rf_train = roc_auc_score(y_train, y_proba_rf_train)

print(f"   Accuracy:  {acc_rf_train:.4f}")
print(f"   Precision: {prec_rf_train:.4f}")
print(f"   Recall:    {rec_rf_train:.4f}")
print(f"   F1-Score:  {f1_rf_train:.4f}")
print(f"   ROC-AUC:   {roc_rf_train:.4f}")

# Test
print(f"\nMÉTRICAS EN TEST:")
acc_rf = accuracy_score(y_test, y_pred_rf_test)
prec_rf = precision_score(y_test, y_pred_rf_test)
rec_rf = recall_score(y_test, y_pred_rf_test)
f1_rf = f1_score(y_test, y_pred_rf_test)
roc_rf = roc_auc_score(y_test, y_proba_rf_test)

print(f"   Accuracy:  {acc_rf:.4f}")
print(f"   Precision: {prec_rf:.4f}")
print(f"   Recall:    {rec_rf:.4f}")
print(f"   F1-Score:  {f1_rf:.4f}")
print(f"   ROC-AUC:   {roc_rf:.4f}")

# Overfitting
diff_f1_rf = abs(f1_rf_train - f1_rf)
print(f"\nANÁLISIS DE OVERFITTING:")
print(f"   Diferencia F1: {diff_f1_rf:.4f}")

if diff_f1_rf < 0.05:
    estado_overfitting_rf = "EXCELENTE - Sin overfitting"
elif diff_f1_rf < 0.10:
    estado_overfitting_rf = "BUENO - Overfitting mínimo"
else:
    estado_overfitting_rf = "ADVERTENCIA - Revisar overfitting"

print(f"   Estado: {estado_overfitting_rf}")

# Matriz de confusión
cm_rf = confusion_matrix(y_test, y_pred_rf_test)
tn_rf, fp_rf, fn_rf, tp_rf = cm_rf.ravel()

print(f"\nMATRIZ DE CONFUSIÓN (TEST):")
print(f"\n                    Predicción")
print(f"                 No Desertor  Desertor")
print(f"Real No Desertor    {tn_rf:6,}      {fp_rf:6,}")
print(f"     Desertor       {fn_rf:6,}      {tp_rf:6,}")

print(f"\nINTERPRETACIÓN:")
print(f"   TN: {tn_rf:,} - No desertores correctos")
print(f"   FP: {fp_rf:,} - Falsa alarma")
print(f"   FN: {fn_rf:,} - Desertores NO detectados (CRÍTICO)")
print(f"   TP: {tp_rf:,} - Desertores detectados")

especificidad_rf = tn_rf / (tn_rf + fp_rf)
vpp_rf = tp_rf / (tp_rf + fp_rf) if (tp_rf + fp_rf) > 0 else 0
vpn_rf = tn_rf / (tn_rf + fn_rf) if (tn_rf + fn_rf) > 0 else 0

print(f"\nMÉTRICAS ADICIONALES:")
print(f"   Especificidad: {especificidad_rf:.4f}")
print(f"   VPP: {vpp_rf:.4f}")
print(f"   VPN: {vpn_rf:.4f}")

# -----------------------------------------------------------------------------
# Importancia
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("IMPORTANCIA DE FEATURES")
print("-"*80)

best_model_rf = grid_rf.best_estimator_
random_forest = best_model_rf.named_steps['clf']

feature_names_rf = []
cat_encoder_rf = best_model_rf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
cat_feature_names_rf = cat_encoder_rf.get_feature_names_out(cat_cols)
feature_names_rf.extend(cat_feature_names_rf)
feature_names_rf.extend(num_cols)

importances_rf = random_forest.feature_importances_

importance_rf_df = pd.DataFrame({
    'Feature': feature_names_rf,
    'Importance': importances_rf
}).sort_values('Importance', ascending=False)

print(f"\nTop 20 FACTORES MÁS DETERMINANTES:")
print(f"\n{'Rank':>5s} {'Feature':50s} {'Importance':>12s}")
print("-" * 70)

for idx, row in importance_rf_df.head(20).iterrows():
    print(f"{importance_rf_df.index.get_loc(idx)+1:>5d} {row['Feature']:50s} {row['Importance']:>12.6f}")

# Concentración
importancia_top5 = importance_rf_df.head(5)['Importance'].sum()
importancia_top10 = importance_rf_df.head(10)['Importance'].sum()

print(f"\nCONCENTRACIÓN DE IMPORTANCIA:")
print(f"   Top 5:  {importancia_top5*100:.2f}%")
print(f"   Top 10: {importancia_top10*100:.2f}%")

# -----------------------------------------------------------------------------
# Visualizaciones
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("GENERANDO VISUALIZACIONES")
print("-"*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Random Forest - Análisis Completo', fontsize=16, fontweight='bold')

# Gráfico 1: Train vs Test
metricas = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
train_vals_rf = [acc_rf_train, prec_rf_train, rec_rf_train, f1_rf_train, roc_rf_train]
test_vals_rf = [acc_rf, prec_rf, rec_rf, f1_rf, roc_rf]

x = np.arange(len(metricas))
width = 0.35

axes[0, 0].bar(x - width/2, train_vals_rf, width, label='Train', color='steelblue', edgecolor='black', alpha=0.8)
axes[0, 0].bar(x + width/2, test_vals_rf, width, label='Test', color='coral', edgecolor='black', alpha=0.8)

axes[0, 0].set_ylabel('Score', fontweight='bold', fontsize=12)
axes[0, 0].set_title('Comparación Train vs Test', fontweight='bold', fontsize=13)
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(metricas, rotation=45, ha='right')
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)
axes[0, 0].set_ylim([0, 1.1])

# Gráfico 2: Matriz confusión
im = axes[0, 1].imshow(cm_rf, cmap='Greens', aspect='auto')
axes[0, 1].set_xticks([0, 1])
axes[0, 1].set_yticks([0, 1])
axes[0, 1].set_xticklabels(['No Desertor', 'Desertor'])
axes[0, 1].set_yticklabels(['No Desertor', 'Desertor'])
axes[0, 1].set_xlabel('Predicción', fontweight='bold', fontsize=12)
axes[0, 1].set_ylabel('Real', fontweight='bold', fontsize=12)
axes[0, 1].set_title('Matriz de Confusión', fontweight='bold', fontsize=13)

for i in range(2):
    for j in range(2):
        axes[0, 1].text(j, i, f'{cm_rf[i, j]:,}',
                       ha="center", va="center", color="black", fontsize=14, fontweight='bold')

plt.colorbar(im, ax=axes[0, 1])

# Gráfico 3: Top 15 features
top_15_rf = importance_rf_df.head(15)
axes[1, 0].barh(range(len(top_15_rf)), top_15_rf['Importance'].values, color='#2ecc71', edgecolor='black', alpha=0.7)
axes[1, 0].set_yticks(range(len(top_15_rf)))
axes[1, 0].set_yticklabels(top_15_rf['Feature'].values, fontsize=9)
axes[1, 0].set_xlabel('Importance', fontweight='bold', fontsize=12)
axes[1, 0].set_title('Top 15 Features', fontweight='bold', fontsize=13)
axes[1, 0].invert_yaxis()
axes[1, 0].grid(axis='x', alpha=0.3)

# Gráfico 4: ROC
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf_test)
axes[1, 1].plot(fpr_rf, tpr_rf, linewidth=3, color='#2ecc71', label=f'ROC (AUC = {roc_rf:.4f})')
axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[1, 1].set_xlabel('Tasa Falsos Positivos', fontweight='bold', fontsize=12)
axes[1, 1].set_ylabel('Tasa Verdaderos Positivos', fontweight='bold', fontsize=12)
axes[1, 1].set_title('Curva ROC', fontweight='bold', fontsize=13)
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(DIRS['visualizaciones'] / 'random_forest_completo.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"   Guardada: random_forest_completo.png")

# -----------------------------------------------------------------------------
# Guardar
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("GUARDANDO RESULTADOS")
print("-"*80)

with open(DIRS['modelos'] / 'random_forest.pkl', 'wb') as f:
    pickle.dump(grid_rf, f)

importance_rf_df.to_csv(DIRS['resultados'] / 'feature_importance_random_forest.csv', index=False)

metricas_rf = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'modelo': 'Random Forest',
    'hiperparametros': {
        'mejor_configuracion': grid_rf.best_params_,
        'n_combinaciones': n_combinaciones_rf
    },
    'metricas_train': {
        'accuracy': float(acc_rf_train),
        'precision': float(prec_rf_train),
        'recall': float(rec_rf_train),
        'f1_score': float(f1_rf_train),
        'roc_auc': float(roc_rf_train)
    },
    'metricas_test': {
        'accuracy': float(acc_rf),
        'precision': float(prec_rf),
        'recall': float(rec_rf),
        'f1_score': float(f1_rf),
        'roc_auc': float(roc_rf)
    },
    'matriz_confusion': {
        'TN': int(tn_rf),
        'FP': int(fp_rf),
        'FN': int(fn_rf),
        'TP': int(tp_rf)
    },
    'overfitting': {
        'diferencia_f1': float(diff_f1_rf),
        'estado': estado_overfitting_rf
    },
    'importancia': {
        'top5_acumulada': float(importancia_top5),
        'top10_acumulada': float(importancia_top10)
    },
    'cv_score': float(grid_rf.best_score_),
    'tiempo_minutos': float(tiempo_rf/60)
}

guardar_resultados_json(metricas_rf, 'metricas_random_forest')

print(f"\nArchivos guardados:")
print(f"   - random_forest.pkl")
print(f"   - metricas_random_forest.json")
print(f"   - feature_importance_random_forest.csv")

# -----------------------------------------------------------------------------
# Resumen
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("RESUMEN - RANDOM FOREST")
print("="*80)

print(f"""
MODELO ENTRENADO Y EVALUADO

CONFIGURACIÓN:
   Grid Search: {n_combinaciones_rf} combinaciones
   Validación cruzada: {N_FOLDS}-fold
   Tiempo: {tiempo_rf/60:.2f} minutos

MEJOR CONFIGURACIÓN:
   {grid_rf.best_params_}

RESULTADOS (TEST):
   F1-Score:  {f1_rf:.4f}
   Precision: {prec_rf:.4f}
   Recall:    {rec_rf:.4f}
   ROC-AUC:   {roc_rf:.4f}

MATRIZ DE CONFUSIÓN:
   TP: {tp_rf:,} | FN: {fn_rf:,}
   
TOP 3 FACTORES:
   1. {importance_rf_df.iloc[0]['Feature']}
   2. {importance_rf_df.iloc[1]['Feature']}
   3. {importance_rf_df.iloc[2]['Feature']}

CONCENTRACIÓN:
   Top 5: {importancia_top5*100:.2f}%

OVERFITTING: {estado_overfitting_rf}

PRÓXIMO PASO:
   Ejecutar src/06_gradient_boosting.py (BLOQUE 5C)
""")

print("="*80)
print("COMPLETADO EXITOSAMENTE")
print("="*80)
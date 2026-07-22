"""
modelamiento_t3.py

Entrega T3 - Implementacion y Resultados
Proyecto: ia-control-calidad-acero
Clasificacion de fallas superficiales en planchas de acero estructural

Este script:
1. Carga y prepara los datos (split estratificado train/test)
2. Entrena 3 modelos: SVM (baseline), Random Forest (principal), XGBoost (comparacion)
3. Entrena una variante adicional con SMOTE (sobre Random Forest) para comparar
   el manejo del desbalance de clases
4. Evalua todos los modelos con F1-macro, balanced accuracy y matriz de confusion
5. Guarda una tabla comparativa final en results/tabla_comparativa.csv

Ejecutar con: python src/modelamiento_t3.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, balanced_accuracy_score, accuracy_score,
                              confusion_matrix, classification_report)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42

# Rutas ancladas a la ubicacion de este archivo (no a la carpeta desde la que se ejecute),
# para que funcione igual usando el boton "Run" de VS Code, la terminal con o sin "cd src",
# u otro editor. Antes se usaba una ruta relativa a la carpeta de trabajo ("../data/..."),
# que fallaba si el script se ejecutaba desde una carpeta distinta a src/.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "steel_plates_faults.csv")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET_COL = "tipo_falla"
ONEHOT_COLS = ["falla_pastry", "falla_z_scratch", "falla_k_scatch", "falla_manchas",
               "falla_suciedad", "falla_abolladuras", "falla_otras"]


def cargar_datos():
    """Carga el dataset y separa variables predictoras (X) de la variable objetivo (y)."""
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COL] + ONEHOT_COLS)
    y = df[TARGET_COL]
    print(f"Dataset cargado: {X.shape[0]} filas, {X.shape[1]} variables predictoras")
    print(f"Distribucion de clases:\n{y.value_counts()}\n")
    return X, y


def dividir_datos(X, y):
    """Split estratificado 80/20, preservando la proporcion de cada clase (clave dado el desbalance)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape[0]} filas | Test: {X_test.shape[0]} filas")
    return X_train, X_test, y_train, y_test


def evaluar_modelo(nombre, modelo, X_test, y_test, escalar_con=None):
    """Calcula F1-macro, balanced accuracy y accuracy; devuelve tambien la matriz de confusion."""
    X_eval = escalar_con.transform(X_test) if escalar_con is not None else X_test
    y_pred = modelo.predict(X_eval)

    f1_macro = f1_score(y_test, y_pred, average="macro")
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(y_test.unique()))

    print(f"\n--- {nombre} ---")
    print(f"F1-macro: {f1_macro:.3f} | Balanced accuracy: {bal_acc:.3f} | Accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {
        "modelo": nombre,
        "f1_macro": round(f1_macro, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "accuracy": round(acc, 4),
    }, cm, y_pred


def graficar_matriz_confusion(cm, labels, titulo, filename):
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(titulo)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{filename}", dpi=120)
    plt.close()


def main():
    X, y = cargar_datos()
    X_train, X_test, y_train, y_test = dividir_datos(X, y)
    labels_ordenadas = sorted(y.unique())

    resultados = []

    # --- 1. Baseline: SVM (kernel RBF) — requiere variables escaladas ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    svm = SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE)
    svm.fit(X_train_scaled, y_train)
    res_svm, cm_svm, _ = evaluar_modelo("SVM (baseline)", svm, X_test, y_test, escalar_con=scaler)
    resultados.append(res_svm)
    graficar_matriz_confusion(cm_svm, labels_ordenadas, "Matriz de Confusión — SVM (baseline)", "cm_svm.png")

    # --- 2. Modelo principal: Random Forest ---
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    res_rf, cm_rf, _ = evaluar_modelo("Random Forest (principal)", rf, X_test, y_test)
    resultados.append(res_rf)
    graficar_matriz_confusion(cm_rf, labels_ordenadas, "Matriz de Confusión — Random Forest", "cm_rf.png")

    # --- 3. Modelo de comparación: XGBoost ---
    y_train_codificado = y_train.astype("category").cat.codes
    y_test_codificado = y_test.astype("category").cat.codes
    categorias = y_train.astype("category").cat.categories

    xgb = XGBClassifier(n_estimators=300, random_state=RANDOM_STATE, eval_metric="mlogloss")
    xgb.fit(X_train, y_train_codificado)
    y_pred_xgb_cod = xgb.predict(X_test)
    y_pred_xgb = categorias[y_pred_xgb_cod]

    f1_xgb = f1_score(y_test, y_pred_xgb, average="macro")
    bal_acc_xgb = balanced_accuracy_score(y_test, y_pred_xgb)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    cm_xgb = confusion_matrix(y_test, y_pred_xgb, labels=labels_ordenadas)
    print(f"\n--- XGBoost (comparación) ---")
    print(f"F1-macro: {f1_xgb:.3f} | Balanced accuracy: {bal_acc_xgb:.3f} | Accuracy: {acc_xgb:.3f}")
    resultados.append({"modelo": "XGBoost (comparación)", "f1_macro": round(f1_xgb, 4),
                        "balanced_accuracy": round(bal_acc_xgb, 4), "accuracy": round(acc_xgb, 4)})
    graficar_matriz_confusion(cm_xgb, labels_ordenadas, "Matriz de Confusión — XGBoost", "cm_xgb.png")

    # --- 4. Random Forest + SMOTE (aplicado solo al train set, despues del split) ---
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=4)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"\nDespués de SMOTE, distribución de train:\n{y_train_sm.value_counts()}")

    rf_smote = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
    rf_smote.fit(X_train_sm, y_train_sm)
    res_rf_smote, cm_rf_smote, _ = evaluar_modelo("Random Forest + SMOTE", rf_smote, X_test, y_test)
    resultados.append(res_rf_smote)
    graficar_matriz_confusion(cm_rf_smote, labels_ordenadas, "Matriz de Confusión — Random Forest + SMOTE", "cm_rf_smote.png")

    # --- Tabla comparativa final ---
    tabla = pd.DataFrame(resultados)
    tabla.to_csv(f"{RESULTS_DIR}/tabla_comparativa.csv", index=False)
    print("\n=== TABLA COMPARATIVA FINAL ===")
    print(tabla.to_string(index=False))

    # --- Importancia de variables (Random Forest, modelo principal) ---
    importancias = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    importancias.to_csv(f"{RESULTS_DIR}/feature_importances.csv")

    plt.figure(figsize=(9, 8))
    importancias.head(15).sort_values().plot(kind="barh", color="teal")
    plt.title("Importancia de variables — Random Forest (top 15)")
    plt.xlabel("Importancia")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/feature_importances.png", dpi=120)
    plt.close()

    # Guardar modelos y objetos necesarios para la etapa de interpretabilidad
    import joblib
    joblib.dump(rf, f"{RESULTS_DIR}/modelo_random_forest.pkl")
    X_test.to_csv(f"{RESULTS_DIR}/X_test.csv", index=False)
    y_test.to_csv(f"{RESULTS_DIR}/y_test.csv", index=False)
    X_train.to_csv(f"{RESULTS_DIR}/X_train.csv", index=False)

    print("\nListo. Resultados guardados en la carpeta results/.")


if __name__ == "__main__":
    main()

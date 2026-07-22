# Clasificación de Fallas Superficiales en Planchas de Acero Estructural mediante Machine Learning

**Entrega T3 — Implementación, Resultados y Presentación Final**
Curso: Aplicaciones de IA en Estructuras — Docente: Ing. Kurt Soncco Sinchi
Universidad Peruana de Ciencias Aplicadas (UPC), Julio 2026

---

## Resumen

Este trabajo aborda la clasificación automática de fallas superficiales en planchas de acero estructural mediante Machine Learning, utilizando el dataset público *Steel Plates Faults* (UCI Machine Learning Repository). Se compararon cuatro configuraciones de modelo — SVM como línea base, Random Forest, XGBoost y Random Forest con remuestreo SMOTE — evaluadas mediante F1-macro y balanced accuracy dado el desbalance de clases identificado en el análisis exploratorio (T2). El modelo **Random Forest con SMOTE** obtuvo el mejor desempeño (**F1-macro = 0.831**), superando al baseline SVM (F1-macro = 0.737). Se aplicaron además técnicas de interpretabilidad (Partial Dependence Plots y SHAP) enfocadas en el par de clases más confundible identificado en el EDA, `Stains` y `Dirtiness`.

**Términos clave:** Machine Learning, control de calidad, acero estructural, clasificación multiclase, Random Forest, SMOTE, interpretabilidad, SHAP.

---

## I. Introducción

El acero estructural es uno de los materiales base más utilizados en la construcción de edificaciones, puentes y naves industriales. Durante su proceso de laminado, las planchas pueden desarrollar defectos superficiales que comprometen su resistencia mecánica y su desempeño estructural. Detectar y clasificar estos defectos a tiempo es fundamental para el control de calidad en la industria siderúrgica, tradicionalmente dependiente de inspección visual humana.

Este trabajo, desarrollado en tres entregas escalonadas (T1, T2 y T3) siguiendo el marco *Veridical Data Science* (VDS) de Yu y Barter [2], tiene como objetivo evaluar si es posible clasificar automáticamente el tipo de falla superficial de una plancha de acero, entre 7 categorías posibles, a partir de indicadores geométricos y de luminosidad extraídos por un sistema de inspección óptica.

---

## II. Metodología

### A. Dataset

Se utilizó el dataset *Steel Plates Faults* [1], desarrollado por el Semeion Research Center (Italia) y publicado en el UCI Machine Learning Repository (DOI [10.24432/C5J88N](https://doi.org/10.24432/C5J88N)). Contiene **1,941 planchas de acero** inspeccionadas, con **27 variables predictoras** numéricas (geometría del defecto, luminosidad, tipo e índices de forma) y una variable objetivo categórica con 7 clases: `Pastry`, `Z_Scratch`, `K_Scatch`, `Stains`, `Dirtiness`, `Bumps` y `Other_Faults`. El dataset no presenta valores nulos ni duplicados (ver [`data/steel_plates_faults.csv`](../data/steel_plates_faults.csv)).

El análisis exploratorio (T2, ver [`notebooks/EDA_T2.ipynb`](../notebooks/EDA_T2.ipynb)) identificó dos hallazgos determinantes para el diseño experimental:
- Un **desbalance de clases** marcado (`Other_Faults` = 34.7% de los casos, `Dirtiness` = 2.8%).
- **Multicolinealidad** fuerte entre varias variables (p. ej. `y_minimo_px` con `y_maximo_px`, r ≈ 1.0).

### B. Preparación de datos

Se realizó una partición train/test (80/20) **estratificada por clase**, para preservar la proporción de cada tipo de falla en ambos conjuntos dado el desbalance observado. Como estrategia complementaria de manejo del desbalance, se generó una variante adicional aplicando **SMOTE** [3] exclusivamente sobre el conjunto de entrenamiento, después de la partición, para evitar fuga de información hacia el conjunto de prueba.

### C. Algoritmos evaluados

Con base en la multicolinealidad detectada, se priorizaron modelos basados en árboles, robustos a este fenómeno frente a modelos lineales:

| Algoritmo | Rol | Justificación |
|---|---|---|
| **Random Forest** | Modelo principal | Robusto a multicolinealidad; feature importance interpretable |
| **XGBoost** | Modelo de comparación | Mayor capacidad predictiva potencial |
| **SVM (RBF)** | Baseline | Más simple, sensible al escalado |

Se descartaron algoritmos de clustering, por tratarse de un problema supervisado con etiquetas conocidas, y redes neuronales profundas, dado el riesgo de sobreajuste con menos de 2,000 observaciones.

### C.1 Código fuente — `src/modelamiento_t3.py`

> El siguiente es el código completo utilizado para el entrenamiento y evaluación de los 4 modelos (SVM, Random Forest, XGBoost, Random Forest+SMOTE). También disponible como archivo ejecutable en [`src/modelamiento_t3.py`](../src/modelamiento_t3.py).

```python
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
DATA_PATH = "../data/steel_plates_faults.csv"
RESULTS_DIR = "../results"

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

```

### C.2 Salida real de la consola al ejecutar `modelamiento_t3.py`

> Esto es exactamente lo que se imprime en pantalla al correr el script — incluye el conteo de filas por clase, el reporte de precisión/recall/F1 por cada una de las 7 fallas para cada modelo, y la tabla comparativa final. Son los mismos números reportados en la Sección III (Resultados).

```text
Dataset cargado: 1941 filas, 27 variables predictoras
Distribucion de clases:
tipo_falla
Other_Faults    673
Bumps           402
K_Scatch        391
Z_Scratch       190
Pastry          158
Stains           72
Dirtiness        55
Name: count, dtype: int64

Train: 1552 filas | Test: 389 filas

--- SVM (baseline) ---
F1-macro: 0.737 | Balanced accuracy: 0.800 | Accuracy: 0.717
              precision    recall  f1-score   support

       Bumps       0.65      0.69      0.67        81
   Dirtiness       0.71      0.91      0.80        11
    K_Scatch       0.99      0.92      0.95        78
Other_Faults       0.76      0.51      0.61       135
      Pastry       0.41      0.81      0.55        32
      Stains       0.67      0.86      0.75        14
   Z_Scratch       0.77      0.89      0.83        38

    accuracy                           0.72       389
   macro avg       0.71      0.80      0.74       389
weighted avg       0.75      0.72      0.72       389


--- Random Forest (principal) ---
F1-macro: 0.819 | Balanced accuracy: 0.789 | Accuracy: 0.799
              precision    recall  f1-score   support

       Bumps       0.78      0.69      0.73        81
   Dirtiness       1.00      0.91      0.95        11
    K_Scatch       0.99      0.91      0.95        78
Other_Faults       0.70      0.85      0.77       135
      Pastry       0.56      0.44      0.49        32
      Stains       1.00      0.86      0.92        14
   Z_Scratch       0.97      0.87      0.92        38

    accuracy                           0.80       389
   macro avg       0.86      0.79      0.82       389
weighted avg       0.81      0.80      0.80       389


--- XGBoost (comparación) ---
F1-macro: 0.813 | Balanced accuracy: 0.805 | Accuracy: 0.802

Después de SMOTE, distribución de train:
tipo_falla
K_Scatch        538
Bumps           538
Other_Faults    538
Z_Scratch       538
Pastry          538
Dirtiness       538
Stains          538
Name: count, dtype: int64

--- Random Forest + SMOTE ---
F1-macro: 0.831 | Balanced accuracy: 0.826 | Accuracy: 0.799
              precision    recall  f1-score   support

       Bumps       0.73      0.72      0.72        81
   Dirtiness       1.00      0.91      0.95        11
    K_Scatch       0.97      0.94      0.95        78
Other_Faults       0.75      0.75      0.75       135
      Pastry       0.57      0.72      0.64        32
      Stains       0.92      0.86      0.89        14
   Z_Scratch       0.92      0.89      0.91        38

    accuracy                           0.80       389
   macro avg       0.84      0.83      0.83       389
weighted avg       0.81      0.80      0.80       389


=== TABLA COMPARATIVA FINAL ===
                   modelo  f1_macro  balanced_accuracy  accuracy
           SVM (baseline)    0.7374             0.7999    0.7172
Random Forest (principal)    0.8188             0.7894    0.7995
    XGBoost (comparación)    0.8131             0.8051    0.8021
    Random Forest + SMOTE    0.8306             0.8257    0.7995

Listo. Resultados guardados en la carpeta results/.
```

### D. Métricas de evaluación

Dado el desbalance de clases, se adoptó **F1-macro** como métrica principal de decisión, ya que promedia el desempeño de las 7 clases sin ponderar por frecuencia. Se reportan también *balanced accuracy* y *accuracy* global como referencia, además de la matriz de confusión para el análisis cualitativo de errores.

### E. Interpretabilidad

Se generaron Partial Dependence Plots (PDP) [4] y explicaciones SHAP [5] enfocadas específicamente en el par de clases `Stains` y `Dirtiness`, identificado en el EDA como el más propenso a confundirse por compartir defectos de bordes difusos.

**Código fuente:** [`src/modelamiento_t3.py`](../src/modelamiento_t3.py) (entrenamiento y evaluación) y [`src/interpretabilidad_t3.py`](../src/interpretabilidad_t3.py) (PDP y SHAP).

### E.1 Código fuente — `src/interpretabilidad_t3.py`

> Código completo utilizado para generar los Partial Dependence Plots y la explicación SHAP. También disponible en [`src/interpretabilidad_t3.py`](../src/interpretabilidad_t3.py).

```python
"""
interpretabilidad_t3.py

Entrega T3 - Interpretabilidad del modelo
Proyecto: ia-control-calidad-acero

Genera:
1. Partial Dependence Plots (PDP) para las variables mas importantes,
   enfocados en distinguir Stains vs Dirtiness (el par mas confundible
   identificado en el EDA de T2).
2. Explicacion SHAP de una prediccion individual, para justificar ante
   un operador de planta por que el modelo clasifico una falla especifica.

Requiere haber ejecutado antes modelamiento_t3.py (usa el modelo y los
datos de test guardados en results/).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.inspection import PartialDependenceDisplay

RESULTS_DIR = "../results"

def cargar_artefactos():
    modelo = joblib.load(f"{RESULTS_DIR}/modelo_random_forest.pkl")
    # Se convierten a float64 porque PartialDependenceDisplay no admite columnas int
    # (evita errores de redondeo implicito al perturbar variables para el PDP)
    X_test = pd.read_csv(f"{RESULTS_DIR}/X_test.csv").astype("float64")
    y_test = pd.read_csv(f"{RESULTS_DIR}/y_test.csv").iloc[:, 0]
    X_train = pd.read_csv(f"{RESULTS_DIR}/X_train.csv").astype("float64")
    return modelo, X_train, X_test, y_test


def generar_pdp(modelo, X_train):
    """PDP sobre las variables mas relevantes para distinguir Stains vs Dirtiness,
    segun lo identificado en el EDA de T2: luminosidad_minima e indice_bordes."""
    variables_foco = ["luminosidad_minima", "indice_bordes", "indice_vacio"]

    fig, ax = plt.subplots(figsize=(14, 4.5))
    PartialDependenceDisplay.from_estimator(
        modelo, X_train, features=variables_foco,
        target="Stains", ax=ax
    )
    plt.suptitle("Partial Dependence — clase 'Stains'")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/pdp_stains.png", dpi=120)
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 4.5))
    PartialDependenceDisplay.from_estimator(
        modelo, X_train, features=variables_foco,
        target="Dirtiness", ax=ax
    )
    plt.suptitle("Partial Dependence — clase 'Dirtiness'")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/pdp_dirtiness.png", dpi=120)
    plt.close()
    print("PDP guardados: pdp_stains.png, pdp_dirtiness.png")


def generar_shap(modelo, X_train, X_test, y_test):
    """Explica una prediccion individual real del conjunto de prueba,
    priorizando un caso verdadero de Stains o Dirtiness si existe."""
    explainer = shap.TreeExplainer(modelo)

    # Buscar un caso real de Stains o Dirtiness en el test set para el ejemplo
    candidatos = X_test[y_test.isin(["Stains", "Dirtiness"])]
    caso = candidatos.iloc[[0]] if len(candidatos) > 0 else X_test.iloc[[0]]
    clase_real = y_test.loc[caso.index[0]] if len(candidatos) > 0 else y_test.iloc[0]

    shap_values = explainer.shap_values(caso)
    clases = modelo.classes_
    idx_clase = list(clases).index(clase_real)

    # Manejo de la forma de salida (varia segun version de shap para multiclase)
    if isinstance(shap_values, list):
        valores_caso = shap_values[idx_clase][0]
    else:
        valores_caso = shap_values[0, :, idx_clase]

    contribuciones = pd.Series(valores_caso, index=X_test.columns).sort_values(key=abs, ascending=False)

    plt.figure(figsize=(9, 6))
    top = contribuciones.head(10).sort_values()
    colors = ["crimson" if v < 0 else "steelblue" for v in top]
    top.plot(kind="barh", color=colors)
    plt.title(f"SHAP — contribución por variable\nCaso real: {clase_real} (top 10 variables)")
    plt.xlabel("Contribución al puntaje de la clase predicha")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/shap_caso_individual.png", dpi=120)
    plt.close()

    pred = modelo.predict(caso)[0]
    print(f"Caso explicado -> clase real: {clase_real} | clase predicha: {pred}")
    print("SHAP guardado: shap_caso_individual.png")

    return clase_real, pred, contribuciones.head(10)


def main():
    modelo, X_train, X_test, y_test = cargar_artefactos()
    generar_pdp(modelo, X_train)
    clase_real, pred, contribuciones = generar_shap(modelo, X_train, X_test, y_test)

    with open(f"{RESULTS_DIR}/resumen_interpretabilidad.txt", "w") as f:
        f.write(f"Caso individual explicado con SHAP:\n")
        f.write(f"  Clase real: {clase_real}\n")
        f.write(f"  Clase predicha por el modelo: {pred}\n\n")
        f.write("Top 10 variables que mas influyeron en la prediccion (SHAP):\n")
        f.write(contribuciones.to_string())

    print("\nListo. Interpretabilidad guardada en results/.")


if __name__ == "__main__":
    main()

```

### E.2 Salida real de la consola al ejecutar `interpretabilidad_t3.py`

```text
PDP guardados: pdp_stains.png, pdp_dirtiness.png
Caso explicado -> clase real: Dirtiness | clase predicha: Dirtiness
SHAP guardado: shap_caso_individual.png

Listo. Interpretabilidad guardada en results/.
```

**Resumen guardado en `results/resumen_interpretabilidad.txt`:**

```text
Caso individual explicado con SHAP:
  Clase real: Dirtiness
  Clase predicha por el modelo: Dirtiness

Top 10 variables que mas influyeron en la prediccion (SHAP):
indice_bordes             0.105721
indice_cuadratura        -0.060388
luminosidad_minima        0.039988
indice_orientacion       -0.031660
x_maximo_px               0.028390
x_minimo_px               0.028058
y_maximo_px               0.027671
longitud_transportador    0.026830
indice_bordes_x          -0.025743
y_minimo_px               0.021194
```

Esta salida confirma, con el caso individual real analizado, que el modelo clasificó correctamente una plancha de la clase `Dirtiness`, y que la variable `indice_bordes` fue la que más contribuyó a esa decisión (0.106), seguida de `indice_cuadratura` (-0.060) y `luminosidad_minima` (0.040) — consistente con el patrón observado en el EDA de T2.

---

## III. Resultados

### A. Comparación de modelos

La Tabla II resume el desempeño de los cuatro modelos evaluados sobre el conjunto de prueba (389 planchas).

**Tabla II. Comparación de modelos**

| Modelo | F1-macro | Balanced Accuracy | Accuracy |
|---|---|---|---|
| SVM (baseline) | 0.737 | 0.800 | 0.717 |
| Random Forest | 0.819 | 0.789 | 0.800 |
| XGBoost | 0.813 | 0.805 | 0.802 |
| **Random Forest + SMOTE** ⭐ | **0.831** | **0.826** | 0.800 |

*(tabla también disponible en [`results/tabla_comparativa.csv`](../results/tabla_comparativa.csv))*

Random Forest con SMOTE obtuvo el mejor F1-macro (0.831) y balanced accuracy (0.826), superando al baseline SVM en 9.4 y 2.6 puntos porcentuales respectivamente, y también al Random Forest sin remuestreo (+1.2 puntos de F1-macro). XGBoost obtuvo un desempeño intermedio, sin justificar su menor interpretabilidad frente a la mejora obtenida.

**Fig. 1. Matriz de confusión — Random Forest + SMOTE (modelo final)**

![Matriz de confusión Random Forest + SMOTE](../results/cm_rf_smote.png)

La matriz de confusión del modelo final muestra que la mayoría de los errores se concentran en `Bumps` y `Other_Faults`, mientras que las clases `K_Scatch`, `Stains` y `Dirtiness` alcanzan un desempeño alto — contrario a lo anticipado inicialmente en el EDA para este último par.

### B. Importancia de variables

**Fig. 2. Importancia de variables — Random Forest (top 15)**

![Importancia de variables](../results/feature_importances.png)

Las variables más relevantes fueron `longitud_transportador`, `log_area` y `area_pixeles`, confirmando que los descriptores de tamaño y geometría del defecto son los principales determinantes de la clasificación, en línea con lo observado en el EDA (T2).

### C. Interpretabilidad: Stains vs. Dirtiness

**Fig. 3. Partial Dependence Plot — clase Dirtiness**

![PDP Dirtiness](../results/pdp_dirtiness.png)

El PDP confirma que la probabilidad de la clase `Dirtiness` aumenta en valores altos de `indice_bordes`, consistente con la hipótesis del EDA de que esta falla, al carecer de un contorno definido, se distingue principalmente por su patrón difuso de bordes más que por su luminosidad.

**Fig. 4. Explicación SHAP de una predicción individual real** (clase verdadera: `Dirtiness`, correctamente clasificada)

![SHAP caso individual](../results/shap_caso_individual.png)

Para el caso individual analizado, la variable `indice_bordes` fue la de mayor contribución positiva a la predicción, seguida por `luminosidad_minima`, validando de forma cuantitativa y a nivel de caso individual el patrón detectado de forma agregada en el EDA y el PDP.

---

## IV. Conclusiones

El modelo **Random Forest combinado con SMOTE** resultó el más adecuado para la clasificación de fallas superficiales en acero estructural dentro del alcance de este trabajo, alcanzando un F1-macro de 0.831 y superando consistentemente al baseline SVM y al Random Forest sin remuestreo. La multicolinealidad identificada en el EDA justificó correctamente la preferencia por modelos de árboles sobre modelos lineales.

Las técnicas de interpretabilidad aplicadas (PDP y SHAP) permitieron explicar, con evidencia cuantitativa y no solo intuición, por qué el modelo distingue clases visualmente similares como `Stains` y `Dirtiness`, cumpliendo el objetivo de que el sistema no solo clasifique sino que también justifique sus decisiones ante un equipo de control de calidad.

**Trabajo futuro:** validar el modelo con datos de una planta real, y extender el análisis de interpretabilidad a la totalidad de los pares de clases más confundibles.

---

## Referencias

[1] M. Buscema, S. Terzi, and W. Tastle, "Steel Plates Faults," UCI Machine Learning Repository, 2010. [Online]. Available: https://doi.org/10.24432/C5J88N

[2] B. Yu and R. Barter, *Veridical Data Science: The Practice of Responsible Data Analysis and Decision Making*. Cambridge, MA: MIT Press, 2024.

[3] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic minority over-sampling technique," *J. Artif. Intell. Res.*, vol. 16, pp. 321–357, 2002.

[4] J. H. Friedman, "Greedy function approximation: A gradient boosting machine," *Ann. Statist.*, vol. 29, no. 5, pp. 1189–1232, 2001.

[5] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2017, pp. 4765–4774.

[6] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2016, pp. 785–794.

[7] Anthropic, "Claude (Sonnet 5)" [Large language model], 2026. [Online]. Available: https://claude.ai

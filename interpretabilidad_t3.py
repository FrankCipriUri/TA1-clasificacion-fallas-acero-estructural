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

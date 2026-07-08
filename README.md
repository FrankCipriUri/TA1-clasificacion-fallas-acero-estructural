# TA1-CLASIFICACIÓN DE FALLAS DE ACERO ESTRUCTURAL

### Clasificación automática de fallas superficiales en planchas de acero estructural mediante Machine Learning, para apoyar el control de calidad en la industria siderúrgica.

**Curso:** Aplicaciones de IA en Estructuras (UPC)
**Docente:** Ing. Kurt Soncco Sinchi
**Entrega:** T1 — Definición del Problema, Datos y Repositorio

## Problema

Clasificar automáticamente el tipo de falla superficial de una plancha de acero
(entre 7 categorías posibles) a partir de indicadores geométricos y de luminosidad
extraídos por un sistema de inspección óptica, para automatizar el control de calidad
del acero estructural antes de su uso en obra.

## Dataset

- **Archivo:** `data/steel_plates_faults.csv`
- **Fuente original:** Semeion Research Center of Sciences of Communication (Italia),
  por encargo del Centro Sviluppo Materiali. Publicado en el UCI Machine Learning
  Repository (DOI: 10.24432/C5J88N).
- **Instancias:** 1,941 planchas de acero inspeccionadas
- **Variables:** 27 predictoras numéricas (geometría, luminosidad, tipo de acero,
  índices de forma) + variable objetivo (`tipo_falla`, 7 categorías, y su versión
  one-hot en 7 columnas adicionales)
- **Tipo de problema:** Clasificación multiclase
- Sin valores nulos. Distribución de clases validada contra la fuente original.

## Distribución de clases (variable objetivo)

| Tipo de falla | N.º de casos | % |
|---|---|---|
| Other_Faults | 673 | 34.7% |
| Bumps | 402 | 20.7% |
| K_Scatch | 391 | 20.1% |
| Z_Scratch | 190 | 9.8% |
| Pastry | 158 | 8.1% |
| Stains | 72 | 3.7% |
| Dirtiness | 55 | 2.8% |

> Nótese el desbalance de clases: se recomienda usar F1-score macro o balanced
> accuracy en T2/T3, en lugar de solo accuracy.

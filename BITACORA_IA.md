# Bitácora de Uso de Inteligencia Artificial (IA)

**Proyecto:** ia-control-calidad-acero — Clasificación de fallas superficiales en planchas de acero estructural
**Curso:** Aplicaciones de IA en Estructuras
**Herramienta utilizada:** Claude (Anthropic), modelo Sonnet 5

En cumplimiento de la política de uso de IA del curso, se documenta a continuación el
historial de prompts, iteraciones y validaciones realizadas con apoyo de un LLM durante
la entrega T1.

## Resumen del proceso

El equipo exploró y descartó dos problemáticas antes de llegar a la definitiva, lo cual
se documenta íntegramente por transparencia, tal como exige el curso.

| Iteración | Problemática evaluada | Resultado |
|---|---|---|
| 1 | Predicción de resistencia a la compresión del concreto (regresión, dataset UCI Concrete) | Descartada — el equipo prefirió un problema de clasificación más orientado a control de calidad |
| 2 | Predicción del grado de daño estructural por sismos en edificaciones (Nepal, dataset Richter's Predictor / DrivenData) | Descartada — el equipo decidió volver a la opción de fallas en acero, más acotada para el alcance de T1 |
| 3 (definitiva) | Clasificación de fallas superficiales en planchas de acero estructural (dataset UCI Steel Plates Faults) | **Seleccionada** |

## Detalle de prompts e iteraciones

1. **Prompt inicial:** solicitud de identificación de un problema de
   ingeniería estructural viable, con datos reales tabulables en CSV.

2. **Iteración — selección de dataset (concreto):** la IA propuso, a partir del Anexo
   de datasets sugeridos por el docente, el dataset de resistencia del concreto (UCI)
   como primera opción por ser tabular y de bajo costo computacional. Se descargó,
   validó (1030 filas, 9 columnas, sin nulos, estadísticos coincidentes con la
   literatura de Yeh, 1998).

3. **Iteración — cambio a daño sísmico:** el equipo solicitó una problemática distinta,
   relacionada con predicción de daño estructural por sismos. La IA identificó y validó
   el dataset de Kathmandu Living Labs / DrivenData (terremoto de Gorkha, Nepal, 2015:
   260,601 filas, 40 columnas, distribución de clases verificada contra la fuente
   oficial) y reescribió el informe con esta problemática.

4. **Iteración — cambio final a fallas en acero:** el equipo decidió retomar la opción
   de clasificación de fallas superficiales en planchas de acero (dataset UCI Steel
   Plates Faults, Semeion Research Center, DOI 10.24432/C5J88N). La IA validó el
   dataset (1941 filas, 34 columnas, distribución de clases exacta: Other_Faults=673,
   Bumps=402, K_Scatch=391, Z_Scratch=190, Pastry=158, Stains=72, Dirtiness=55),
   tradujo las columnas al español..

5. **Iteración — marco VDS/PCS:** se solicitó el análisis bajo los principios de
   Predictibilidad, Computabilidad y Estabilidad (Yu & Barter, 2024). Se propuso el
   contenido de cada principio aplicado al dataset final, incluyendo la observación
   sobre desbalance de clases y la recomendación de usar F1-macro/balanced accuracy en
   lugar de accuracy para T2/T3 — comentario que el docente destacó positivamente.

## Evidencia de originalidad y fuentes primarias

Toda afirmación técnica del informe fue contrastada contra la fuente primaria real del
dataset (UCI Machine Learning Repository / Semeion Research Center, DOI 10.24432/C5J88N).

## Responsabilidad y validación (Accountability)

El equipo es responsable de la precisión final del análisis. Antes de avanzar a T2 se
validará independientemente con pandas: tipos de dato, ausencia de nulos, rangos de cada
variable y la distribución de la variable objetivo (`tipo_falla`), y se auditará
cualquier código generado con apoyo de IA en los notebooks de limpieza y modelado,
corrigiendo errores lógicos o de sintaxis que se encuentren.

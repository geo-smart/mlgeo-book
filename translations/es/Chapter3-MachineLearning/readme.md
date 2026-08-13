# Panorama del capítulo

## Capítulo 3: Machine learning clásico en las geociencias

Este capítulo cubre el *machine learning* clásico (aprendizaje automático clásico, CML) para las geociencias: modelos que aprenden de tablas de características (variables explicativas) en lugar de formas de onda o imágenes crudas. El aprendizaje automático clásico se construye rápido, cuesta poco ejecutarlo y es fácil de interrogar. Eso lo convierte en el lugar adecuado para adquirir los hábitos que se trasladan al aprendizaje profundo: modelos de referencia, divisiones honestas de los datos y una evaluación acorde a cómo se usará el modelo en realidad.

### El arco del capítulo

1. **Conceptos** (3.1) — la taxonomía de la supervisión del entrenamiento: aprendizaje supervisado, no supervisado, semisupervisado, autosupervisado, por refuerzo y activo, y dónde aparece cada uno en las geociencias.
2. **Clasificación y regresión** (3.2) — los dos tipos de problema supervisado, un primer flujo de trabajo de extremo a extremo y la división entrenamiento/validación/prueba.
3. **Agrupamiento** (*clustering*) (3.3) — descubrimiento no supervisado de estructura: métricas de distancia, k-means desde cero, diagnósticos de silueta y del codo, agrupamiento jerárquico y un ejercicio de sismicidad volcánica.
4. **Clasificación binaria** (3.4) — detección evento-contra-ruido, comparación de clasificadores, las métricas que importan cuando las clases están desbalanceadas (precisión, exhaustividad, curvas PR frente a ROC) y dos perillas para tratar el desbalance (pesos de clase, desplazamiento del umbral).
5. **Clasificación multiclase** (3.5) — cuatro tipos de fuente sísmica, matrices de confusión por clase, ROC uno-contra-el-resto y el ejercicio de la tabla de clasificación del curso.
6. **Regresión logística desde cero** (3.6) — la única lección donde se abre la caja negra: la función de pérdida, el descenso de gradiente, la diferenciación automática con PyTorch y una verificación de calibración — diagramas de confiabilidad y el score de Brier — sobre las probabilidades predichas.
7. **Árboles, bosques y potenciación** (3.7) — árboles de decisión, regresión con bosques aleatorios, importancia de características y sus trampas (importancia por permutación, dependencia parcial, características correlacionadas), y el *gradient boosting* (potenciación de gradiente) como el estándar moderno para datos tabulares.
8. **Entrenamiento robusto** (3.8) — validación cruzada para datos correlacionados: por qué las divisiones aleatorias mienten en series autocorrelacionadas; divisiones temporales, por grupos (sitio, evento) y espaciales dejando un conglomerado fuera; StratifiedGroupKFold para historiales de casos pequeños y desbalanceados; modelos de referencia de persistencia e intervalos de confianza por *bootstrap* sobre los scores.
9. **Aprendizaje por ensambles** (3.9) — votación, *bagging*, potenciación y apilamiento (*stacking*), con la dispersión del voto del ensamble como primera estimación de la incertidumbre epistémica y la exhaustividad por clase en la comparación de modelos.
10. **Qué fue del AutoML** (3.10) — una breve historia de la búsqueda automática de modelos, las piezas que sobrevivieron (optimización de hiperparámetros con Optuna, buenos valores por defecto del *gradient boosting*) y un ejercicio de evaluación crítica de código de modelado generado por IA.

La reducción de dimensionalidad (PCA, t-SNE) se cubre en el capítulo 2.12 y aquí se usa como paso de preprocesamiento, sin volver a enseñarse.

### El hilo de la evaluación honesta

Una sola disciplina atraviesa cada cuaderno de este capítulo:

- **Primero los modelos de referencia.** Antes de cualquier modelo, establezca qué logra un predictor trivial: la clase mayoritaria en clasificación, la media histórica en regresión. Un modelo que no supera al modelo de referencia no aprendió nada.
- **Nunca evalúe sobre los datos de entrenamiento.** La calidad del modelo se mide sobre datos que el modelo no ha visto. La validación cruzada ocurre dentro del conjunto de entrenamiento; el conjunto de prueba se toca una sola vez.
- **Las divisiones deben respetar la estructura de los datos.** Las series autocorrelacionadas y los datos espaciales agrupados requieren divisiones temporales, por bloques o por grupos (3.8).
- **Los conjuntos de prueba ocultos existen.** El instructor conserva variantes regeneradas de los conjuntos de datos del curso con semillas privadas. Los scores ajustados contra un conjunto de prueba público mostrarán una brecha en el oculto.

La lección 3.5 lo lleva a la práctica con una **tabla de clasificación del curso** (*leaderboard*): cada estudiante entrena el clasificador de su elección sobre una división canónica de un conjunto real de fuentes sísmicas, envía sus predicciones por *pull request* y la integración continua lo califica contra el conjunto de prueba público y contra el oculto.

### Herramientas

El capítulo usa `scikit-learn` como caballo de batalla, `lightgbm` y el *gradient boosting* por histogramas de scikit-learn para árboles potenciados, `optuna` para la búsqueda de hiperparámetros y `pytorch` en 3.6 para introducir la diferenciación automática. Los conjuntos de datos del curso provienen del paquete `mlgeo_synth` (generadores sintéticos con motivación física), del repositorio de datos del curso y de un archivo curado en Zenodo de eventos sísmicos del Noroeste del Pacífico estadounidense.

### Resultados de aprendizaje

Al final de este capítulo, usted podrá:

- Plantear un problema geocientífico como clasificación, regresión o agrupamiento, y elegir un primer modelo apropiado.
- Establecer modelos de referencia y evaluar modelos con métricas adecuadas al problema, incluidas las clases desbalanceadas.
- Diseñar divisiones de datos y esquemas de validación cruzada que respeten la correlación temporal y espacial.
- Entrenar, ajustar y comparar ensambles de árboles y otros modelos clásicos, y reportar los resultados con honestidad.
- Leer un guion de modelado generado por una máquina y encontrar sus fallas.

### Tareas

- **Tarea**: una tarea calificada (un problema de clasificación de geoquímica de roca total) que cubre preparación de datos, PCA, agrupamiento y comparación de modelos.
- **Hito del proyecto final**: una guía de hito (3.20) que aplica estos métodos al conjunto de datos de su propio proyecto.

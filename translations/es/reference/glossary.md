# Glosario

Términos usados a lo largo del libro, desde los dos lados de su público: vocabulario de *machine learning* para estudiantes de geociencias, y vocabulario de geociencias y de procesamiento de señales para estudiantes de cómputo.

Cada entrada conserva el término en inglés — es el que usted encontrará en la literatura, en la documentación de las bibliotecas y en el código — seguido de su forma en español entre paréntesis, siguiendo el [glosario de la edición en español](https://github.com/geo-smart/mlgeo-book/blob/main/translations/GLOSSARY_es.md). Las entradas están en orden alfabético del inglés, para que coincidan con la edición original.

```{glossary}

Ablation (ablación)
: Experimento que elimina un componente de un modelo o de un flujo de trabajo (una característica, un término de la pérdida, una capa) y reentrena, de modo que la diferencia de puntaje mide lo que ese componente aporta. El cuaderno 4.7 ablaciona así el término físico de la pérdida de una PINN.

Agent (agente)
: Un modelo de lenguaje de gran escala (LLM) envuelto en un ciclo que puede invocar herramientas (ejecutar código, leer archivos, buscar), observar los resultados y volver a actuar hacia un objetivo, en vez de producir una sola respuesta de texto. El capítulo 6 construye y evalúa agentes para tareas de investigación.

AI-ready data (datos listos para IA)
: Datos organizados de modo que un modelo pueda consumirlos directamente y quien los lea pueda confiar en ellos: unidades y muestreo consistentes, procedencia y licencias documentadas, etiquetas limpias y formatos legibles por máquina. El capítulo 2 convierte descargas crudas en conjuntos de datos listos para IA.

Baseline (modelo de referencia)
: El modelo creíble más simple para una tarea — persistencia o climatología para pronóstico, la clase mayoritaria para clasificación, la regresión lineal para datos tabulares. Todo modelo complejo debe superar el modelo de referencia para justificar su complejidad.

Calibration (calibración)
: La concordancia entre la confianza declarada de un modelo y la realidad: entre las predicciones hechas con 80 % de probabilidad, cerca del 80 % deberían ser correctas, y los intervalos de predicción del 90 % deberían cubrir cerca del 90 % de los desenlaces. La calibración se mide (diagramas de confiabilidad, cobertura de los intervalos), nunca se supone.

Coda (coda)
: En sismología, la cola de un registro sísmico después de las llegadas principales P y S: ondas dispersadas cuya amplitud decae gradualmente con el tiempo. La forma de la coda ayuda a distinguir tipos de fuente (la coda de una explosión difiere de la de un sismo).

Corner frequency (frecuencia de esquina)
: La frecuencia a la que el espectro de amplitud de un sismo pasa de plano a decreciente. Escala de manera inversa con la duración de la ruptura, así que los sismos más grandes tienen frecuencias de esquina más bajas; es una característica estándar para caracterizar fuentes sísmicas.

Data leakage (fuga de datos)
: Cualquier vía por la que información no disponible al momento de predecir llega al modelo durante el entrenamiento — ajustar un escalador sobre el conjunto completo antes de particionarlo, muestras de prueba correlacionadas con muestras de entrenamiento, valores futuros que informan predicciones del pasado. La fuga infla puntajes que después se derrumban con datos genuinamente nuevos. Vea también *leakage (spatial and temporal)* y, para el término no relacionado del procesamiento de señales, *spectral leakage*.

Deep ensemble (ensamble profundo)
: Varias copias de la misma red entrenadas desde semillas aleatorias distintas. Promediar sus predicciones suele mejorar la exactitud, y su desacuerdo sobre una muestra estima la incertidumbre epistémica (cuaderno 4.5).

Dimensionality reduction (reducción de dimensionalidad)
: Comprimir muchas características en pocas preservando la estructura, ya sea para visualizar los datos (PCA, t-SNE, UMAP) o para alimentar a un modelo con una representación más pequeña. El cuello de botella de un autocodificador es una forma aprendida de esto.

Epistemic vs. aleatoric uncertainty (incertidumbre epistémica frente a aleatoria)
: La incertidumbre epistémica viene de lo que el modelo no sabe — muy pocos datos, condiciones no vistas — y se encoge conforme crecen los datos; los ensambles y el *MC dropout* la estiman. La incertidumbre aleatoria es la aleatoriedad del proceso o de la medición misma (ruido del sensor) y no se encoge con más datos de entrenamiento.

Eval set (conjunto de evaluación)
: Una colección curada de casos de prueba con respuestas conocidas y una regla de calificación automática, usada para medir un LLM o un sistema de agentes en lugar de confiar en impresiones. El cuaderno 6.3 construye uno a partir de datos sintéticos con verdad de referencia conocida.

Expected calibration error (ECE) (error de calibración esperado)
: Un resumen en un solo número de un diagrama de confiabilidad: la brecha promedio entre la confianza declarada y la exactitud observada, ponderada por cuántas predicciones caen en cada intervalo de confianza. Se calcula junto con los diagramas de confiabilidad en los cuadernos 3.6 y 4.5.

Feature engineering (ingeniería de características)
: Construir entradas informativas para el modelo a partir de datos crudos usando conocimiento del dominio: razones STA/LTA, estadísticos espectrales, medias móviles, curtosis. El *machine learning* clásico (capítulo 3) depende de ella; las redes profundas, en cambio, aprenden muchas características directamente de los datos crudos.

Flicker noise (ruido de parpadeo)
: Ruido cuya potencia crece hacia las bajas frecuencias (aproximadamente como 1/f), común en las series de posición GNSS y en la instrumentación electrónica. A diferencia del ruido blanco, no se promedia hasta desaparecer con registros más largos, así que sesga las estimaciones de tendencia y de incertidumbre si se modela como blanco.

[Git](https://git-scm.com)
: El sistema de control de versiones usado en todo el curso para registrar, comparar y compartir el historial del código y del texto.

[GitHub](https://github.com)
: El servicio de alojamiento donde viven los repositorios del curso, los conjuntos de datos y los proyectos estudiantiles, construido alrededor de Git más las *issues* y los *pull requests*.

Grouped cross-validation (validación cruzada por grupos)
: Validación cruzada que mantiene en el mismo pliegue todas las muestras que comparten un grupo — la misma estación, el mismo sitio de campo o el mismo sismo. Los puntajes miden entonces la generalización a grupos nuevos en lugar de la interpolación dentro de los ya conocidos (sección 3.8).

Heteroscedastic noise (ruido heterocedástico)
: Ruido cuya magnitud varía entre muestras — una red mixta de instrumentos de laboratorio y de campo, o estaciones con condiciones de sitio distintas. Cuando se conoce el nivel de ruido de cada muestra, las funciones de pérdida ponderadas por la varianza recuperan la exactitud que se pierde al ignorar esos metadatos (cuaderno 4.5).

Hidden test set (conjunto de prueba oculto)
: Datos reservados que nunca se inspeccionan ni se tocan durante el desarrollo del modelo y que se califican una sola vez, al final. La tabla de clasificación de pronóstico del curso (4.10) usa uno; tocarlo repetidamente lo convierte en un conjunto de validación.

Instrument response (respuesta instrumental)
: La función de transferencia de un sensor y de su sistema de registro, que lleva el movimiento real del suelo a las cuentas registradas. Los registros sísmicos y geodésicos deben corregirse por ella antes de que las amplitudes tengan unidades físicas.

Interpolation vs extrapolation (interpolación frente a extrapolación)
: Predecir dentro de la región cubierta por los datos de entrenamiento, frente a predecir fuera de ella. La mayoría de los modelos aprendidos se degradan bruscamente, y a veces en silencio, al extrapolar; la distribución de entrenamiento es un contrato, y el diseño de la partición (3.8) más las comprobaciones fuera de rango (4.5) permiten saber si un puntaje habla de un régimen o del otro.

Jupyter notebook (cuaderno de Jupyter)
: Un documento que mezcla código ejecutable, sus salidas y texto narrativo; el formato de la mayoría de los capítulos de este libro.

Leakage (spatial and temporal) (fuga espacial y temporal)
: Los dos sabores de fuga que dominan en los datos geocientíficos. Las particiones aleatorias de datos autocorrelacionados colocan casi duplicados a ambos lados de la partición: píxeles o estaciones vecinas (espacial), ventanas de tiempo traslapadas o contiguas (temporal). Las soluciones son particiones espaciales en bloques o con zonas de amortiguamiento, y particiones que respeten el orden temporal.

Linear probe (sonda lineal)
: Un clasificador formado por un codificador preentrenado congelado más una pequeña cabeza lineal entrenada. Mide lo que las características preentrenadas cargan por sí solas; el *fine-tuning* (ajuste fino), en cambio, también actualiza los pesos del codificador. El cuaderno 4.6 compara una sonda lineal contra el entrenamiento desde cero.

MASE
: Error absoluto medio escalado (*mean absolute scaled error*): el error absoluto medio de un pronóstico dividido entre el de un modelo de referencia ingenuo (persistencia o ingenuo estacional) sobre la misma serie. Un MASE menor que 1 supera al modelo de referencia; mayor que 1, el modelo pierde frente a él (cuaderno 4.10).

MC dropout (Monte Carlo dropout)
: Ejecutar muchas veces una red ya entrenada dejando el *dropout* activo al momento de predecir. Cada pasada muestrea una subred ligeramente distinta, y la dispersión de las salidas aproxima la incertidumbre epistémica con un solo modelo entrenado. Se compara cara a cara con un ensamble profundo en el cuaderno 4.5.

[MyST](https://mystmd.org)
: *Markedly Structured Text*, la variante de Markdown y el sistema de construcción con los que este libro está escrito y publicado.

Physics-informed neural network (PINN) (red neuronal informada por la física)
: Una red entrenada con el residuo de una ecuación de gobierno (ley de conservación, ecuación de difusión) añadido a la pérdida junto al desajuste con los datos, de modo que las predicciones se ven atraídas hacia soluciones físicamente consistentes (cuaderno 4.7).

[pixi](https://pixi.sh)
: El gestor de paquetes y entornos usado para instalar la pila de software del curso. Resuelve paquetes del ecosistema conda en un archivo de bloqueo, de modo que cada estudiante y la CI construyen el entorno idéntico.

[pooch](https://www.fatiando.org/pooch/)
: Una pequeña biblioteca de Python que descarga un archivo de datos desde una URL, lo guarda en caché localmente y verifica su suma de verificación SHA256, de modo que quede demostrado que el análisis corrió sobre el archivo previsto.

Spectral leakage (fuga espectral)
: En procesamiento de señales, el desparramo de energía de una frecuencia hacia los intervalos de frecuencia vecinos cuando la ventana analizada no contiene un número entero de ciclos. Aplicar una ventana de suavizado (*tapering*) lo reduce. No tiene relación con *data leakage*.

STA/LTA
: Promedio de corto plazo sobre promedio de largo plazo de la amplitud de la señal: una razón móvil que se dispara cuando llega un transitorio sobre el ruido de fondo. Es el disparador clásico para detectar sismos en datos sísmicos continuos, y una característica construida estándar.

Supervised, unsupervised, and self-supervised learning (aprendizaje supervisado, no supervisado y autosupervisado)
: El aprendizaje supervisado ajusta las entradas a etiquetas provistas por humanos; el aprendizaje no supervisado encuentra estructura (agrupamientos, representaciones de baja dimensión) sin etiquetas; el aprendizaje autosupervisado fabrica etiquetas a partir de los datos mismos — enmascarar parte de la entrada y predecirla — de modo que el preentrenamiento puede usar archivos sin etiquetar de tamaño ilimitado (cuaderno 4.6).

Tolerance-based reproducibility (reproducibilidad basada en tolerancia)
: Declarar reproducido un resultado cuando una reejecución coincide dentro de una tolerancia numérica declarada, en lugar de bit a bit. La aritmética de punto flotante, el orden de ejecución en paralelo y las diferencias de hardware hacen de la igualdad exacta la prueba equivocada para los flujos científicos (capítulo 5).

Transformer
: La arquitectura construida con capas apiladas de autoatención y codificaciones posicionales, introducida por Vaswani et al. (2017). Sustenta los modelos de lenguaje de gran escala y los sistemas de pronóstico actuales; el cuaderno 4.4 construye uno pequeño a partir de sus piezas.

[Zarr](https://zarr.dev)
: Un formato de almacenamiento para arreglos N-dimensionales fragmentados y comprimidos, diseñado para que los almacenes de objetos en la nube puedan servir lecturas parciales en paralelo. La elección común para datos geocientíficos en malla de gran tamaño, junto a NetCDF.

```

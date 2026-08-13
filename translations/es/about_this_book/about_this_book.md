# Machine Learning en las Geociencias

El marco **GeoS**cience **MA**chine Learning **R**esources and **T**raining (GeoSMART) ofrece una ruta educativa en cómputo científico de código abierto, teoría general de ML, herramientas y despliegue.

Este libro acompaña el curso Machine Learning in the Geosciences (ESS 469/569) de la Universidad de Washington. El libro, los tutoriales y las tareas viven en este único repositorio; los estudiantes ejecutan los cuadernos localmente o en el servicio de nube de su elección.

Instructores:

- Marine Denolle (mdenolle@uw.edu)
- Akshay Mehra (akmehra@uw.edu)

Este proyecto cuenta con el apoyo del equipo GeoSMART (Stefan Todoran, Nicoleta Cristea, Anthony Arendt, Scott Henderson, Ziheng Sun, Yiyu Ni, Akash Kharita).

## Panorama

El curso introduce el *machine learning* (aprendizaje automático) en las geociencias, las bases del cómputo y la metodología aplicada de ML. Trabaja con conjuntos de datos canónicos y de actualidad en sismología, oceanografía, criósfera, ciencias planetarias, geología y geodesia. Los métodos enseñados incluyen agrupamiento no supervisado, regresión logística, bosque aleatorio (*random forest*), máquinas de vectores de soporte y aprendizaje profundo (*deep learning*) con PyTorch.

El curso descansa sobre tres pilares, más una cuarta capa que atraviesa todo en la edición 2026:

1. **Datos listos para IA**: convertir observaciones geocientíficas crudas en conjuntos de datos de los que un modelo pueda aprender.
2. **Machine learning clásico**: métodos basados en características, entrenados y evaluados con honestidad.
3. **Aprendizaje profundo**: redes neuronales en PyTorch, del perceptrón a las arquitecturas modernas.
4. **Trabajar con IA agéntica**: los estudiantes de 2026 escriben código junto a asistentes de IA que leen repositorios, ejecutan código y proponen cambios. El curso trata esto como una habilidad que se enseña, no como un atajo que se vigila. La evaluación crítica de la salida de la IA, la traducción de resultados para audiencias distintas y la articulación del impacto posterior son habilidades calificadas, al mismo nivel que la exactitud (*accuracy*) del modelo.

# Para quién es este libro

Estudiantes de doctorado y de últimos años de licenciatura en las geociencias y en ingenierías y ciencias de la computación afines, más los profesores, postdocs, científicos de laboratorio y personal de programas que los forman o los contratan. Quien lo completa puede llevar un flujo de datos geocientíficos crudo y desordenado hasta un modelo defendible, reproducible y evaluado con honestidad — y puede decir exactamente qué partes hizo un asistente de IA.

# Resultados de aprendizaje

Al final del libro, los estudiantes podrán:

| # | Resultado | Nivel de Bloom | Dónde en el libro | Artefacto calificado |
|---|---------|-------------|-------------------|-----------------|
| 1 | Describir los usos canónicos del ML en las geociencias (descubrimiento, automatización, procesamiento de señales, emulación, pronóstico) y emparejar familias de métodos con tipo de datos, tamaño de muestra y pregunta | Comprender | readmes de capítulo, 3.1 | cuestionarios cronometrados en la plataforma del curso |
| 2 | Construir entornos de cómputo reproducibles (Git, pixi, cuadernos) y correr el mismo flujo en una laptop, en HPC o en una instancia de nube | Aplicar | capítulo 1, 5.4 | Tarea 1, banco de trabajo |
| 3 | Transformar flujos de datos crudos diversos — series de tiempo de sensores desde muestreo diario hasta 100 Hz, imágenes geoespaciales y campos en malla, observaciones tabulares y puntuales — en conjuntos de datos listos para IA, y reparar patologías reales de instrumentos (huecos, deriva, errores de tiempo, muestreo irregular, ruido en las etiquetas) | Analizar / Crear | capítulo 2 | ejercicios del capítulo 2, sección de datos del proyecto final |
| 4 | Aplicar transformadas estadísticas y de procesamiento de señales (filtrado, transformadas de Fourier y ondículas, remuestreo, reducción de dimensionalidad) y predecir su efecto sobre lo que un modelo puede aprender después | Aplicar / Evaluar | 2.6–2.12 | ejercicios del capítulo 2 |
| 5 | Diseñar la evaluación antes que el modelo: elegir modelos de referencia del dominio, construir particiones sin fuga de datos (temporales, espaciales, por grupos) y reportar métricas con incertidumbre | Evaluar / Crear | capítulo 3 (validación cruzada espacial y por grupos en 3.8), tabla de clasificación (*leaderboard*) | tabla de clasificación (3.5, 4.10), tareas |
| 6 | Construir y entrenar modelos clásicos y profundos (regresión, bosques, *boosting*; MLP, CNN, RNN, *transformer*, autocodificador) en scikit-learn y PyTorch, y diagnosticar corridas de entrenamiento buenas y fallidas | Aplicar / Crear / Analizar | capítulos 3–4 | tareas de ML clásico y aprendizaje profundo, laboratorio 4.5 |
| 7 | Explotar el conocimiento físico como activo de datos: generar datos sintéticos de entrenamiento y de referencia a partir de modelos físicos, incorporar restricciones físicas en pérdidas y arquitecturas, y validar contra una verdad conocida | Crear / Evaluar | 2.10, 4.7, `mlgeo_synth` | laboratorio 4.5, ejercicio de conjunto de evaluación 6.3 |
| 8 | Cuantificar la incertidumbre predictiva (*bootstrap* y ensambles, *MC dropout* en inferencia, salidas por cuantiles y distribucionales), comprobar la calibración y juzgar cuándo un modelo extrapola más allá de su distribución de entrenamiento | Evaluar | a lo largo de los capítulos 3–4 (3.8–3.9, 4.5, 4.10) | ejercicios de los capítulos 3–4, proyecto final |
| 9 | Versionar y rastrear datos, modelos y experimentos de modo que otro científico reejecute el flujo y obtenga los mismos números | Aplicar / Evaluar | capítulo 5 | repositorio del proyecto final (30%) |
| 10 | Evaluar agentes de IA y salidas de LLM como instrumentos científicos: escribir una especificación de tarea, construir un conjunto de evaluación con verdad de referencia, calificar salidas y analizar modos de falla | Evaluar / Crear | capítulo 6 | ejercicio de conjunto de evaluación 6.3, agente de revisión final |
| 11 | Integrar la asistencia de IA a la investigación conservando la propiedad intelectual del trabajo: declarar el uso, verificar las salidas y defender cada decisión metodológica sin el asistente | Aplicar / Evaluar | 1.8, 6.4 | declaraciones de uso, presentaciones |
| 12 | Traducir el mismo resultado para audiencias distintas y valorar los usos y consecuencias posteriores de un modelo desplegado | Crear / Evaluar | capítulo 7 | entregables del proyecto final (7.1, 7.2) |

Los conceptos de visualización de datos se introducen y se usan a lo largo del libro.

# Prerrequisitos

**Prerrequisitos** (numeración de la Universidad de Washington): MATH 207 y MATH 208, o MATH 307 o 308, o AMATH 351 o 352, CS160 o CS163, o permiso del instructor. En otras instituciones: álgebra lineal, ecuaciones diferenciales básicas y un primer curso de programación.

**Habilidades recomendadas**: conocimiento de Python, métodos numéricos básicos, cursos introductorios de ciencias de la Tierra. Ofrecemos repasos de cómputo como parte del curso.

# Temario

- **Parte I: GeoDatos listos para IA**: los datos geocientíficos, sus modalidades y dimensiones, características básicas, extracción de características, reducción de dimensionalidad y cómo dar formato a un conjunto de datos listo para IA a partir de datos geocientíficos.
- **Parte II: Machine Learning clásico**: entrenamiento de modelos, evaluación, valoración de la generalización y buenas prácticas para el entrenamiento confiable de algoritmos clásicos tras la ingeniería de características (por ejemplo, K-means, bosque aleatorio, k-NN).
- **Parte III: Aprendizaje profundo**: conceptos fundamentales del aprendizaje profundo — perceptrones y redes totalmente conectadas, redes convolucionales y recurrentes, un *transformer* pequeño para pronóstico de secuencias, autocodificadores y redes neuronales informadas por la física — más práctica de entrenamiento: optimización, regularización, diagnóstico de corridas fallidas e incertidumbre en las salidas del modelo.

Los capítulos posteriores extienden los pilares: flujos reproducibles en la era de los agentes (capítulo 5), construcción y evaluación de agentes de IA (capítulo 6), y casos de uso, traducción para audiencias e impacto posterior (capítulo 7).

# Construcción de habilidades técnicas

A lo largo del curso, los estudiantes construyen habilidades en *shell*, control de versiones con git y GitHub, programación en Python, cómputo de alto rendimiento y visualización de datos en Python.

- _Shell_: se introduce temprano en el curso, se usa según haga falta.
- _Control de versiones_: se introduce temprano y se usa en cada clase.
- _Programación en Python_: se introduce progresivamente. Detallamos el uso de numpy, (geo)pandas y scikit-learn, con PyTorch como marco de aprendizaje profundo.
- _Visualización en Python_: se introduce temprano con Matplotlib y Plotly, se usa en cada clase de Python.
- _Cómputo de alto rendimiento_: se usa en la segunda mitad del curso y durante el proyecto final.
- _Asistentes de IA agéntica_: se introducen en el capítulo 1 (vea la [política de uso de IA del curso](../Chapter1-GettingStarted/1.8_ai_in_your_workflow.md)) y se usan, con declaración, en todo el curso.

# Lecturas: de los artículos a un agente de revisión

Las lecturas asignadas siguen un arco de cuatro etapas a lo largo del trimestre:

1. **Revisión de literatura asistida por IA**: los estudiantes dirigen a un asistente de IA en una revisión de literatura sobre un tema asignado y verifican ellos mismos cada cita.
2. **Anatomía de los buenos artículos científicos**: disecar artículos ejemplares — qué hace defendibles los métodos, honestas las figuras y sustentadas las afirmaciones.
3. **Construya sus propios estándares de calidad**: mediante discusión en clase, cada estudiante escribe una rúbrica de calidad explícita en su género declarado — artículo de revista o entregable para partes interesadas, en correspondencia con las dos pistas del proyecto final.
4. **Construya un agente de revisión previa al envío**: los estudiantes convierten su rúbrica en un agente de revisión y lo prueban con la maquinaria de evaluación del capítulo 6 (especificación de tarea, conjunto de evaluación, análisis de fallas) contra artículos con fortalezas y defectos conocidos.

La programación semana a semana del arco vive en el temario del curso.

# Infraestructura del curso

Este libro contiene todos los tutoriales y tareas. Los estudiantes trabajan en VS Code o JupyterLab con un asistente de IA agéntica, guardan su trabajo en GitHub y gestionan los entornos de software con [pixi](https://pixi.sh). Para construir el libro localmente:

```
pixi install
pixi run build
```

Cada estudiante crea un repositorio personal del curso llamado `MLGEO2026_UWNETID`, copia a él los archivos de entorno de este libro y mantiene ahí, bajo control de versiones, sus tareas y su trabajo de proyecto.

# Licencias

El texto y las figuras de este libro están bajo licencia [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/) (CC-BY-4.0). El código, incluido el código fuente de los cuadernos, está bajo la [licencia MIT](https://opensource.org/license/mit). Puede reutilizar y adaptar ambos con atribución.

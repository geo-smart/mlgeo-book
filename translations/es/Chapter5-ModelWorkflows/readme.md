# Capítulo 5: Flujos de trabajo, reproducibilidad y rigor en la era de los agentes

Los capítulos 1 a 4 le enseñaron a construir modelos y a evaluarlos con justicia. Este capítulo le enseña a operar el conjunto completo, de modo que cualquiera — incluida su versión de dentro de seis meses, e incluido un agente de IA trabajando en su repositorio — pueda reejecutarlo y obtener la misma respuesta.

El momento importa. En 2026, buena parte del código de un proyecto de investigación lo redactan agentes. Eso aumenta el valor de las prácticas de este capítulo en lugar de reducirlo: un agente puede producir un análisis verosímil, equivocado e irreproducible más rápido de lo que usted alcanza a leerlo. Los entornos fijados, las transformaciones programadas, los experimentos con seguimiento y la integración continua son la forma de mantener el control sobre trabajo que usted no escribió.

## Qué contiene este capítulo

1. **[5.1 Reproducibilidad](5.1_reproducibility.md)** — Reproducibilidad frente a replicabilidad, y la pila de la reproducibilidad: entornos fijados, semillas, datos crudos inmutables, contenedores y verificaciones ejecutables. La propia construcción de este libro es el caso de estudio.
2. **[5.2 Seguimiento de experimentos](5.2_experiment_tracking.ipynb)** — Un laboratorio práctico. Usted construye un rastreador de experimentos mínimo en unas 30 líneas y lo usa para correr un pequeño estudio de hiperparámetros; luego ve qué agregan MLflow y Weights & Biases sobre las mismas ideas.
3. **[5.3 Control de versiones de datos y modelos](5.3_data_model_versioning.md)** — Por qué Git falla con los datos, qué hacen DVC y sus pares, la práctica mínima viable con sumas de verificación, y cómo versionar modelos con fichas de modelo. Además: cómo copiar el patrón de CI de este libro a su repositorio de proyecto.
4. **[5.4 Cómputo más allá de la laptop](5.4_compute_beyond_laptop.md)** — Cuándo salirse de la laptop, la escalera de opciones desde el HPC departamental hasta la nube comercial, disciplina de costos y acceso a datos optimizado para la nube.

## Resultados de aprendizaje

Al final de este capítulo, usted podrá:

- enunciar la diferencia entre reproducibilidad y replicabilidad, y decir cuál de las dos pone a prueba una verificación dada;
- fijar un entorno para que una persona colaboradora (o CI, o un agente) reejecute su código con las mismas versiones de bibliotecas;
- hacer seguimiento de experimentos de modo que cada número reportado pueda rastrearse hasta una corrida, un *commit* y una versión del conjunto de datos;
- decidir cuándo un proyecto necesita más cómputo que una laptop, y elegir la opción adecuada más barata;
- configurar CI que ejecute sus cuadernos en cada *pull request*.

Estas habilidades alimentan directamente la rúbrica del proyecto final ([criterios de reproducibilidad y documentación](../about_this_book/1.10_MLGEO_FinalProject.md)).

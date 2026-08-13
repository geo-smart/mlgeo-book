# El proyecto MLGeo

Esta página recorre las etapas del diseño de un proyecto de *machine learning* (aprendizaje automático) en las geociencias, y señala los capítulos que enseñan cada etapa.

## 1. Encuadre el proyecto

* Motive la necesidad del *machine learning* en su proyecto científico.

Haga una revisión de la literatura sobre las preguntas científicas abiertas y las soluciones propuestas hasta ahora. ¿Cuáles serían los pasos para resolver el problema a mano? ¿Cuáles son las limitaciones de las soluciones actuales? ¿Un algoritmo de ML nuevo sería lo bastante generalizable como para aplicarse a otros 10 o más problemas de investigación? ¿Cuál es el potencial a 5-10 años de ese problema en particular, dadas las tecnologías nuevas, las nuevas instalaciones de investigación o su relevancia social? ¿Existen problemas comparables en los que las herramientas puedan reutilizarse?

* ¿En qué estado están los datos?

¿Hay muchos datos, y datos etiquetados? ¿Hay experiencia humana disponible? ¿Será un problema de aprendizaje supervisado o no supervisado? ¿Se puede acceder a los datos desde archivos de acceso abierto que cumplan los principios FAIR (localizables, accesibles, interoperables, reutilizables)? ¿Cuál sería su DOI? ¿Existen regulaciones locales o acuerdos de recolección que restrinjan cómo pueden compartirse los datos?


## 2. Organice el proyecto — Capítulo 1

Inicie un repositorio de GitHub con un README.md, cree una especificación de entorno (pixi o un archivo YML) y use nombres de archivos y carpetas legibles por humanos y por máquinas. Asegúrese de que el nombre del proyecto no se haya usado antes.

## 3. Descarga de datos — Capítulo 1

Enumere los datos, su información, sus etiquetas y su procedencia (incluida la accesibilidad desde archivos de datos de acceso abierto). ¿Qué tan grandes son los datos? ¿Qué formato de datos sería óptimo para leerlos desde varios lenguajes (Python, C, R, MATLAB, Julia, etc.)? ¿Puede almacenar metadatos? ¿Cómo se comporta en entrada/salida?

¿Los datos son geoespaciales o son series de tiempo?

Encuentre una plataforma de cómputo apropiada para el almacenamiento y la entrada/salida de los datos (cómputo en la nube, clúster Linux institucional, etc.).

Cree un cuaderno de Jupyter que documente la descarga y el almacenamiento de los datos.

## 4. Preparación de los datos — Capítulo 2

* **Explore los datos**

Cree un cuaderno de Jupyter para la exploración preliminar de los datos. Documente:
- El nombre y el tipo de dato
- El ruido: qué tipo de ruido hay (estocástico, valores atípicos, huecos en los datos, etc.)
- La distribución de los datos: gaussiana, uniforme, logarítmica, etc.
- Las etiquetas de los datos (o atributos objetivo)


Visualice un subconjunto de los datos.

Estudie las correlaciones básicas entre atributos.

¿Cómo resolvería el problema a mano con estos datos?

Identifique transformaciones que puedan ser útiles (como STFT, CWT, PCA).

Guarde las gráficas y los cuadernos preliminares. Documente los hallazgos.

* **Acondicionamiento previo de los datos — Capítulo 2**

Copie los datos y trabaje sobre esas copias.

Escriba funciones para todas las transformaciones de datos, de modo que puedan invocarse automáticamente (y depurarse con facilidad). Esas funciones se usarán para los conjuntos de entrenamiento, validación y prueba.

**Limpie los datos**: corrija o elimine los valores atípicos, rellene los valores faltantes (con cero, la media o la mediana), o descarte datos (cuando hay demasiados huecos, por ejemplo).
Guarde la copia limpia de los datos en un archivo distinto.

Tenga cuidado con los datos sintéticos. Los datos aleatorios de juguete (ruido salido de un generador, sin física detrás) dicen poco: un algoritmo que funciona sobre ellos puede comportarse de manera completamente distinta con observaciones reales, así que evítelos para cualquier cosa más allá de una prueba de humo. Los datos sintéticos motivados físicamente y con verdad de referencia documentada son otra cosa. Como usted conoce la respuesta verdadera, son admisibles para el desarrollo de métodos, la comparación de desempeño y los conjuntos de prueba ocultos. La sección 2.10 y el paquete `mlgeo_synth` del libro generan datos de este tipo. Para la ciencia misma, prefiera datos recolectados del mundo real.

* **Preparación de características — Capítulo 2**

**Descarte** los atributos que no sirvan para la tarea.

**Transforme** las características (por ejemplo, con la STFT).

Explore características rápidas y prometedoras (por ejemplo, la PGA para movimientos del suelo).

**Escale** las características para estandarizarlas o normalizarlas. En la mayoría de los casos, los algoritmos de ML no funcionan bien sin normalizar las características o los datos de entrada. El escalamiento no es un requisito, pero tiende a mejorar el comportamiento del entrenamiento.
+ *Escalamiento mín-máx*: quita el valor mínimo y luego normaliza por el valor máximo de la distribución, de modo que las amplitudes queden entre 0 y 1. Es apropiado cuando las características son números positivos. La función incorporada de scikit-learn ``sklearn.preprocessing.MinMaxScaler()`` calcula:

<code>X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))</code>

<code>X_scaled = X_std * (max - min) + min</code>
Scikit-learn tiene funciones incorporadas para hacer el escalamiento.

+ *Estandarización*: quita la media y divide por la desviación estándar. La distribución de salida no tiene cotas. Es más estable que el escalamiento mín-máx porque es menos sensible a los valores atípicos. La función incorporada de scikit-learn es ````sklearn.preprocessing.StandardScaler()````

Hay otras maneras de normalizar las características o los datos de entrada.
https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing



## 5. Reducción de dimensionalidad — Capítulo 2

Explore las formas posibles de reducir la dimensión de los datos (PCA, ICA).

Documente la transformación de los datos con cuadernos. Reasigne los atributos y las etiquetas de los datos en las nuevas coordenadas.

## 6. Diseño del modelo — Capítulos 3 y 4

Encuentre el ***modelo de referencia*** (*baseline*) que el proyecto de ML debe superar. Como mínimo, su algoritmo de ML tiene que superar el modelo de referencia *aleatorio*; si no, hay problemas en el diseño del modelo o en los datos de entrada.

Pruebe varios algoritmos de modelado. El *teorema de no hay almuerzo gratis* (Wolpert 1995): no existe el mejor algoritmo de aprendizaje en general, solo un algoritmo muy exacto sobre un conjunto de datos dado.

El modelo debe tener la **complejidad mínima** necesaria para **minimizar el error esperado del modelo**.


## 7. Entrenamiento del modelo — Capítulos 3 y 4

Separe los datos en tres conjuntos: un conjunto de entrenamiento para ajustar el modelo, un conjunto de validación para afinar los hiperparámetros y guiar el diseño del modelo, y un conjunto de prueba que se toca una sola vez, al final, para reportar el desempeño.

Diseñe la partición antes de preocuparse por sus proporciones. Los datos geocientíficos están correlacionados en el tiempo y en el espacio, así que una partición aleatoria suele filtrar información: muestras de la misma tormenta, del mismo sismo o de la misma estación caen a ambos lados de la frontera, y el puntaje de prueba se vuelve optimista. Ajuste la partición a la estructura de correlación de los datos:

- *Partición temporal*: entrene con el pasado y pruebe con el futuro, cuando las muestras están ordenadas en el tiempo.
- *Partición espacial*: reserve regiones enteras, con una zona de amortiguamiento, cuando las muestras están autocorrelacionadas espacialmente.
- *Partición por grupos*: mantenga todas las muestras de un mismo evento, estación o sitio del mismo lado de la partición.

La [sección 2.13](../Chapter2-DataManipulation/2.13_MLready_data.ipynb) convierte las particiones de referencia y los controles de fuga en parte de la definición de un conjunto de datos listo para IA; la [sección 3.8](../Chapter3-MachineLearning/3.8_robust_training.ipynb) enseña la validación cruzada consciente de la fuga de datos, incluidas sus variantes espacial y por grupos. Una vez que el diseño de la partición está a salvo de fugas, las proporciones son secundarias (70/15/15 y 60/20/20 son comunes). La validación cruzada sobre la porción de entrenamiento y validación — con pliegues que respeten la misma estructura temporal, espacial o de grupos — estima el error esperado del algoritmo de aprendizaje y su dispersión.

Guarde los resultados intermedios cuando sea posible.

Guarde las semillas del generador de números aleatorios para poder reproducir los resultados.

Evite escribir su propia biblioteca de código casera. Use fuentes confiables.

Documente bien, sobre todo cuando trabaje en grupo.

Empiece con algo más pequeño que la corrida final. Se sugiere no usar más del 25 % de los recursos disponibles en el primer diseño del modelo.

Elija una medida de desempeño.




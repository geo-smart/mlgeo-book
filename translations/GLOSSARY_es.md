# Glosario ES — terminología de la edición en español

> **Tabla trilingüe:** [`GLOSSARY.md`](GLOSSARY.md) da, para cada término en
> inglés, el término francés *y* el término español lado a lado, indicando qué
> palabras conserva en inglés cada comunidad. Esa es la tabla para leer de un
> idioma a otro. Esta página sigue siendo la referencia del traductor
> hispanohablante: contiene el detalle de uso y el historial de las filas en
> disputa.

## Qué es esta página, y qué no

Es una **guía de uso**, no una autoridad por encima de los capítulos. Describe
lo que escriben quienes investigan en español, contexto por contexto, para que
quien traduce elija la forma adecuada en el lugar adecuado. Cuando un capítulo
y esta tabla no coinciden, casi siempre es el capítulo el que reporta el uso
real y la tabla la que hay que corregir: así se revisó la mayoría de las filas
de abajo. Abra una issue `[translation] term: …` en lugar de reescribir la
prosa de un capítulo para que calce con una fila.

Una palabra inglesa no equivale a una palabra española. Varios términos de este
libro cubren dos o tres conceptos distintos — *workflow*, *pipeline*,
*repository*, *cluster*, *notebook*, *build* — y fundirlos en una sola palabra
destruye una distinción que la lectora necesita. Esos términos llevan **una
fila por sentido**.

«EN conservado» = el término inglés permanece en la prosa (uso establecido),
glosado en español en su primera aparición por capítulo. No es una tolerancia
concedida a regañadientes: es lo que la comunidad dice y escribe. La regla de
fondo es **glosar una vez por capítulo y luego mantener la consistencia dentro
del capítulo**. Español neutro panregional: se fija una variante y se sostiene.
Las filas marcadas ⚑ están en disputa — abra una issue `[translation] term: …`.
Las filas marcadas **error** no son preferencias: equivocarse cambia lo que
dice la frase.

## Registro ML — la decisión central

**Primera aparición:** «aprendizaje automático (*machine learning*, **ML**)».
**Después:** «aprendizaje automático» en la prosa explicativa y **ML** en la
prosa técnica compacta, en diagramas, tablas, comparaciones y referencias al
ecosistema.

El español **no** necesita el nombre inglés del campo como forma por defecto de
su prosa, a diferencia del francés. Pero **sí debe enseñar la sigla ML**, que
es la que aparece en el código, en las diapositivas, en los títulos de cursos y
en los artículos; presentarla como jerga ajena deja a quien estudia sin puente
hacia la literatura. El uso institucional también es mixto: la página del ICATE
(CONICET) escribe «Aprendizaje Automático (Machine Learning en inglés)».

**Las dos ediciones difieren a propósito.** La edición francesa usa *machine
learning* / **ML** como forma de prosa por defecto para el campo y reserva la
familia « apprentissage… » para los paradigmas con nombre; ver
[`GLOSSARY_fr.md`](GLOSSARY_fr.md). La española hace lo contrario. No es una
inconsistencia que haya que normalizar: es lo que hace cada comunidad de
investigación, y la evidencia de cada lado está en su propia página.

| English | Español | ¿EN conservado? | Nota |
|---|---|---|---|
| machine learning | **aprendizaje automático** (*machine learning*, **ML**) | la sigla **ML**, sí, y debe enseñarse explícitamente | ver la sección de arriba. «ML» en prosa técnica compacta; «aprendizaje automático» en prosa explicativa |
| training set | conjunto de entrenamiento | no | nunca «set de entrenamiento» |
| validation set | conjunto de validación | no | |
| test set | conjunto de prueba | no | ⚑ «conjunto de test» circula; se fija «prueba» |
| hidden test set | conjunto de prueba oculto | no | vocabulario del curso |
| data leakage | fuga de datos | no | **error** si va sin calificar: siempre «de datos». Sin calificativo choca con la fuga espectral |
| spectral leakage | fuga espectral | no | **error** si va sin calificar: siempre «espectral». Dispersión de energía entre bins de frecuencia por el truncamiento de la ventana — sin relación con la fuga de datos |
| overfitting | sobreajuste | no | |
| underfitting | subajuste | no | |
| cross-validation | validación cruzada | no | |
| grouped cross-validation | validación cruzada por grupos | no | |
| fold | pliegue | ⚑ | alternativa: «partición de validación» |
| baseline | *baseline* (modelo de referencia) | **sí**, glosado una vez por capítulo | quienes investigan dicen *baseline*; después, cualquiera de las dos formas, sostenida en todo el capítulo |
| persistence (forecast) | persistencia | no | |
| feature | «característica» o «variable explicativa» según la disciplina | *feature* como puente al código | no imponer una sola forma entre estadística y aprendizaje profundo. **Error:** nunca «atributo», reservado para *attribute*. Identificadores (`n_features`) sin traducir |
| label | etiqueta | no | |
| supervised learning | aprendizaje supervisado | no | |
| unsupervised learning | aprendizaje no supervisado | no | |
| self-supervised learning | aprendizaje autosupervisado | no | |
| reinforcement learning | aprendizaje por refuerzo | no | |
| active learning | aprendizaje activo | no | |
| clustering | «agrupamiento» o *clustering*, presentados juntos | **sí**, a elección | *clustering* es común; presentarlo una vez — «el agrupamiento (*clustering*)» — y no volver a glosarlo en cada aparición |
| classification | clasificación | no | |
| regression | regresión | no | |
| random forest | bosque aleatorio | no | |
| gradient boosting | — | **sí** (*gradient boosting*) | glosado «potenciación de gradiente» una vez |
| neural network | red neuronal | no | |
| deep learning | aprendizaje profundo | *deep learning* en una primera mención o en un título | paradigma con nombre: en prosa corriente, «aprendizaje profundo» |
| convolutional neural network (CNN) | red neuronal convolucional | sigla CNN conservada | |
| recurrent neural network (RNN) | red recurrente | sigla RNN conservada | |
| transformer | — | **sí** (*transformer*) | ⚑ «transformador» circula; se conserva EN |
| autoencoder | autocodificador | ⚑ *autoencoder* tolerado | |
| dropout | — | **sí** (*dropout*) | glosado «apagado aleatorio de neuronas» una vez |
| MC dropout | — | **sí** | |
| deep ensemble | ensamble profundo | no | «ensamble», no «ensemble» |
| calibration | calibración | no | |
| reliability diagram | diagrama de confiabilidad | no | ⚑ «fiabilidad» (España) vs «confiabilidad» (América); se fija confiabilidad |
| Brier score | score de Brier | ⚑ | `mean((p − o)²)`, `o ∈ {0,1}`; menor es mejor. O «puntaje de Brier»; el uso profesional dice «score». Lección 3.6 |
| coverage (of intervals) | cobertura (empírica) de los intervalos | no | |
| expected calibration error (ECE) | error de calibración esperado | sigla ECE conservada | promedio ponderado de `abs(exactitud(b) − confianza(b))` sobre los bins de confianza `b`; el número depende del binning, hay que declararlo. Lección 4.5 |
| skill (forecast) | — | **sí** (*skill*) glosado «habilidad de pronóstico» | uso operativo. `1 − score/score_ref`: un **número** calculado contra un pronóstico de referencia (persistencia, climatología). Nombrar siempre la referencia |
| lead time | horizonte de pronóstico | no | ⚑ «plazo de pronóstico» circula |
| epistemic / aleatoric uncertainty | incertidumbre epistémica / aleatoria | no | |
| hyperparameter | hiperparámetro | no | |
| epoch | época | no | |
| batch | lote | *batch* tolerado | |
| learning rate | tasa de aprendizaje | no | |
| loss function | función de pérdida | no | **error**: no confundir con la función objetivo ni con el costo. Es el error en **un** ejemplo |
| objective (function) | función objetivo | no | **error** si se usa como sinónimo de *loss*: es la pérdida agregada **más** los términos de regularización, lo que el optimizador minimiza |
| cost | costo (o «coste») | no | la consecuencia real de un error (una alerta perdida, una evacuación innecesaria). La palabra no está prohibida; la confusión sí |
| split criterion (árboles) | criterio de partición | no | Gini, entropía y error cuadrático son criterios de partición, no «funciones de costo» |
| gradient descent | descenso de gradiente | no | |
| accuracy | exactitud | no | **error: nunca «precisión»**. `(TP+TN)/(TP+TN+FP+FN)`, mientras que *precision* = `TP/(TP+FP)`. Primera lección con fórmula: 3.4 |
| precision | precisión | no | **error** si se usa para *accuracy*. `TP/(TP+FP)`. Lección 3.4 |
| recall | «exhaustividad», «sensibilidad» o *recall* según el dominio | **sí** en prosa técnica | `TP/(TP+FN)`. ⚑ relajado: la fórmula fija la cantidad, el dominio elige la palabra — recuperación de información y ML dicen «exhaustividad»; medicina, epidemiología y detección esperan «sensibilidad»; en la práctica hispanohablante de ML también se dice *recall*. Enunciar la fórmula en la primera aparición y sostener una sola palabra por capítulo. Lección 3.4 |
| F1 | puntaje F1 / score F1 | sigla F1 conservada | `2·P·R/(P+R)`, media armónica de precisión y exhaustividad. ⚑ hereda la duda «puntaje» vs «score» de la fila *score* |
| eval set (agents) | conjunto de evaluación | no | vocabulario del capítulo 6 |
| agent | agente | no | |
| large language model (LLM) | modelo de lenguaje de gran escala | sigla LLM conservada | |
| prompt | — | **sí** (*prompt*) | glosado «instrucción» una vez |
| pretraining / fine-tuning | preentrenamiento / ajuste fino | *fine-tuning* tolerado glosado | |
| reproducibility | reproducibilidad | no | |
| sampling rate | tasa de muestreo | no | |
| aliasing | solapamiento espectral | *aliasing* tolerado glosado | |
| leaderboard | tabla de clasificación | *leaderboard* tolerado glosado | vocabulario del curso |
| data card | ficha de datos | no | |
| computer | computadora | no | se fija la variante americana; nunca alternar con «ordenador» |

## Una palabra inglesa, varios sentidos — la tabla de contexto

Estos términos no tienen *una* traducción. La columna «sentido» es la que
decide, y es la primera que hay que leer. Estas filas reemplazan equivalencias
únicas anteriores («cluster → conglomerado», «notebook → cuaderno», «recall →
exhaustividad») que los capítulos contradecían — y los capítulos tenían razón.

| Término inglés | Sentido | Forma española | Qué evitar en ese sentido |
|---|---|---|---|
| workflow | proceso de investigación, nivel conceptual | «metodología», «procedimiento» | «flujo de trabajo» por reflejo |
| workflow | cómputo ejecutable y reproducible | «flujo de trabajo computacional» | — es la forma natural en español y se mantiene |
| workflow | archivo de automatización de CI | «workflow de GitHub Actions» o «flujo de trabajo de GitHub Actions»; la ruta `.github/workflows/` **nunca se traduce** | «archivo» a secas, que colisiona con el sentido de preservación |
| pipeline | transformaciones ordenadas de señal o datos | «cadena de procesamiento» | «flujos de procesamiento» genérico en contexto de ingesta de señal |
| pipeline | encadenamiento de modelado (sin objeto de API) | «cadena de modelado» | |
| pipeline | objeto de scikit-learn (preprocesamiento + estimador) | *pipeline*; «un *pipeline* de scikit-learn (`Pipeline`)» en la primera aparición | traducir el identificador `Pipeline` |
| repository | proyecto Git o GitHub con control de versiones | «repositorio Git», «repositorio GitHub» | **error**: «archivo» |
| archive | depósito de preservación y difusión (Zenodo, HAL) | «archivo», «archivo de datos» | usarlo para un proyecto Git activo |
| data repository | portal de datos gestionado | «repositorio de datos» | elegir por la ortografía inglesa en vez de por la función |
| object storage | almacenamiento de objetos (S3, MinIO, buckets) | «almacenamiento de objetos» | «archivo», salvo que la preservación sea de verdad la función |
| notebook | documento interactivo, prosa orientada a investigación | «notebook de Jupyter (cuaderno)» en la primera aparición, luego «notebook» | imponer «cuaderno» como única forma escrita; traducir `.ipynb` o un nombre de archivo |
| cluster | grupos resultantes de un algoritmo de agrupamiento | «grupo», «conglomerado» o «clúster» según la comunidad | ⚑ «conglomerado» como forma única panregional: no viaja igual en todas partes |
| cluster | infraestructura de cómputo | «clúster de cómputo» | forzar «conglomerado» en contexto de cómputo |
| cloud | infraestructura remota | «nube», «computación en la nube» | traducir nombres de producto (AWS, Google Cloud, Azure) |
| HPC | cómputo paralelo en centro de cómputo | «HPC (cómputo de alto rendimiento)» una vez, luego HPC | traducir la sigla |
| job | ejecución enviada a un planificador o a CI | «trabajo», «tarea» o *job* según la cultura del planificador | traducir campos y comandos del planificador; explicar el vocabulario una vez |
| build | etapa de CI, empaquetado | *build* | forzar «compilación» donde se habla de CI |
| build | producto publicado del libro | «compilación» | |
| dataset | conjunto de datos en prosa | «conjunto de datos» | expulsar *dataset* de una frase que nombra un objeto (`Dataset` de Xarray, un conjunto de HF) |
| benchmark | conjunto de datos de referencia | «conjunto de referencia» | «banco de pruebas» para un simple conjunto de datos; «punto de referencia», que se lee como un hito |
| benchmark | protocolo o suite de comparación | «banco de pruebas» | |
| inference | fase de despliegue / predicción | «inferencia» para la fase, «predicción» para el acto concreto | usar «inferencia» en sentido estadístico de estimación sin aclararlo |

## Identificadores — nunca se traducen

Nombres de funciones, clases y argumentos (`Pipeline`, `.fit()`, `DataLoader`,
`StandardScaler`); nombres de archivo y rutas (`environment.yml`, `pixi.lock`,
`.github/workflows/`, `*.ipynb`); claves de configuración YAML/TOML/JSON;
nombres de columnas que usa el código (`target`, `label`, `uncertainty`,
`pick_weight`); salidas ejecutadas y reportes de modelo. El patrón de escritura
es «la variable objetivo `target`», nunca una columna renombrada que rompe el
notebook.

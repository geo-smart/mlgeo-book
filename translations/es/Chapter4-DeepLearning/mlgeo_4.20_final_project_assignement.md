# Hito del proyecto final: aprendizaje profundo sobre su conjunto de datos listo para IA

**Objetivo**: Demostrar que usted puede implementar, entrenar, diagnosticar y evaluar críticamente modelos de aprendizaje profundo sobre su propio conjunto de datos listo para IA, compararlos con el aprendizaje automático clásico y entregar software reproducible.

Todo lo que se exige abajo se enseña con código funcional en este capítulo. Cuando un requisito nombre un cuaderno, reutilice el patrón de ese cuaderno sobre sus propios datos.

---

## 1. Preparación y exploración del conjunto de datos (10%)

- **Uso de datos listos para IA (4%)**: Use el conjunto de datos listo para IA que preparó antes, con un preprocesamiento consistente en todos los modelos. Describa las entradas, su significado físico, sus modalidades y sus dimensiones.
- **Análisis exploratorio de datos (3%)**: Visualizaciones y resúmenes de la distribución de los datos y de su estructura temporal y espacial.
- **Planteamiento del problema (3%)**: Defina la tarea (regresión o clasificación) y dé forma a los datos para cada arquitectura (ventaneo para modelos de secuencias, remodelado para CNN).

---

## 2. Comparación con el aprendizaje automático clásico (10%)

- **Modelos de referencia (5%)**: Reporte sus resultados de aprendizaje automático clásico del hito anterior (bosque aleatorio, *gradient boosting* o similares). Trabajo nuevo mínimo; el objetivo es tener una línea de referencia.
- **Comparación de desempeño (5%)**: Compare los modelos clásicos y los profundos con las mismas métricas sobre las mismas divisiones.

---

## 3. Exploración de arquitecturas de modelos (30%)

- **Al menos tres arquitecturas (8%)**: Implemente y entrene al menos tres de las siguientes: MLP, CNN 1D o 2D, LSTM, codificador *transformer*, autoencoder + cabeza de clasificación o regresión. Las cinco se construyen en los cuadernos 4.1-4.6. La U-Net es opcional, no obligatoria. Justifique cada elección frente a sus datos y al tipo de problema. Escriba cada arquitectura con las dimensiones de sus capas y sus funciones de activación.
- **Exploración de hiperparámetros (7%)**: Explore de manera sistemática la tasa de aprendizaje, los tamaños de las capas y otros hiperparámetros, siguiendo el laboratorio del cuaderno 4.5. Documente cada experimento; una tabla de corridas le gana a la prosa dispersa.
- **Estudio de ablación (5%)** — obligatorio: Quite un componente de su mejor modelo (un bloque de capas, el *dropout*, la normalización por lotes, un grupo de características o un término de la pérdida) y reporte el cambio en el desempeño. Basta con una ablación hecha con cuidado; enuncie qué le dice sobre el componente.
- **Pérdida informada por la física (4%)**: Agregue un término de pérdida informado por la física o consciente del dominio allí donde su problema lo admita (el patrón del cuaderno 4.7). Si su problema no lo admite, explique por qué en un párrafo breve; una respuesta negativa bien argumentada obtiene el crédito completo.
- **Innovación (6%)**: Arquitecturas híbridas, funciones de pérdida personalizadas o aumento de datos específico de las geociencias.

---

## 4. Evaluación del desempeño (20%)

- **Evaluación cuantitativa (6%)**: Métricas para todos los modelos: exactitud, precisión, exhaustividad (*recall*), F1, RMSE o medidas específicas del dominio. Los problemas multiclase reportan precisión y exhaustividad por clase. Enuncie el optimizador, la tasa de aprendizaje y el tamaño de lote de cada modelo entrenado.
- **Generalización y pruebas fuera de distribución (7%)**: Evalúe sobre datos no vistos o fuera de distribución y discuta el sobreajuste o el subajuste.
- **Experimento elegido (4%)** — elija UNO:
  - *Preentrenamiento frente a entrenar desde cero*: preentrene un codificador sobre sus datos sin etiquetas (autoencoder o reconstrucción enmascarada, el patrón del cuaderno 4.6), luego entrene una sonda lineal sobre el codificador congelado — o haga ajuste fino de todo el codificador — con una fracción pequeña de datos etiquetados, y compárelo con entrenar desde cero sobre la misma fracción. Diga cuál de las dos cosas hizo; una sonda mantiene fijos los pesos del codificador, el ajuste fino los actualiza.
  - *Incertidumbre por ensamble profundo*: entrene su mejor modelo con 5 semillas aleatorias, reporte el desempeño de la media del ensamble y la varianza de las predicciones por muestra, y muestre en qué muestras el ensamble discrepa (el patrón del cuaderno 4.5).
- **Visualización de resultados (3%)**: Matrices de confusión, gráficas de pérdida frente a época, mapas de error o equivalentes.

---

## 5. Entrega de software y calidad del código (15%)

- **Práctica estándar de entrenamiento (7%)**: Código modular, un cuaderno por sección clara. Aborde: (1) preparación de los datos con la descripción de entrenamiento, validación y prueba, (2) arquitectura y diseño del modelo, (3) estrategia de entrenamiento (tamaño de lote, optimizador, programador) con curvas de aprendizaje, (4) evaluación y generalización.
- **Guardado de resultados (4%)**: Guarde los pesos del modelo, las bitácoras de entrenamiento y las métricas de desempeño en archivos CSV/JSON incluidos en el repositorio.
- **Calidad del código y documentación (4%)**: Legible, comentado, reproducible. El README del repositorio indica cómo correr los cuadernos y en qué orden.

---

## 6. Reporte e interpretación (10%)

- **Comunicación científica (3%)**: Un reporte claro y conciso con figuras y tablas apropiadas.
- **Aportes al dominio (2%)**: Qué significan los resultados para el problema geocientífico: relevancia física, limitaciones de los datos, aplicaciones potenciales.
- **Apéndice de diagnóstico de entrenamiento (5%)** — obligatorio: Curvas de aprendizaje de su modelo final Y de al menos una corrida fallida o defectuosa (pérdida que diverge, sobreajuste, una tasa de aprendizaje que avanzó a paso de tortuga). Diagnostique la corrida fallida en 2-3 oraciones usando el vocabulario del cuaderno 4.5. Las corridas fallidas son evidencia de trabajo sistemático, no algo que esconder.

---

## 7. Consideraciones computacionales y éticas (5%)

- **Reporte de cómputo (3%)**: Tiempo de entrenamiento, hardware usado y huella de memoria de cada modelo, y cómo el costo de cómputo influyó en sus decisiones.
- **Ética y declaración del uso de IA (2%)**: Reflexione sobre los sesgos de sus datos y sobre la transparencia de las predicciones. Incluya una declaración obligatoria de un párrafo sobre el uso de IA: qué asistentes de IA o herramientas de generación de código usó, para qué partes del trabajo y cómo verificó su salida. Usar herramientas de IA está permitido; no declararlas, no.

---

**Total: 100%**

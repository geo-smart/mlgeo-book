# Panorama del capítulo

## Capítulo 4: Aprendizaje profundo

Este capítulo enseña el aprendizaje profundo (*deep learning*) construyéndolo pieza por pieza, en PyTorch, sobre datos de geociencias. Cada arquitectura se implementa, se entrena y se diagnostica en un cuaderno que puede ejecutarse en una computadora portátil. Los modelos son deliberadamente pequeños; las ideas no.

### Mapa del capítulo

1. **El perceptrón** (4.0)
   - Una sola neurona artificial, implementada desde cero
   - La regla de aprendizaje del perceptrón y sus límites
   - El descenso de gradiente comparado con mínimos cuadrados ordinarios

2. **Una primera red neuronal** (4.1)
   - Los cinco pasos de todo script de entrenamiento: conjunto de datos, modelo, pérdida, optimizador, bucle de entrenamiento
   - Clasificación multiclase de fuentes sísmicas a partir de características tabulares
   - Lectura de curvas de aprendizaje

3. **Perceptrones multicapa** (4.2)
   - Profundidad, *dropout* (apagado aleatorio de neuronas) y normalización por lotes
   - Guardar, crear puntos de control (*checkpoints*) y restaurar modelos
   - PyTorch comparado con el MLPClassifier de scikit-learn

4. **Redes neuronales convolucionales** (4.3)
   - Convolución y *kernels* (núcleos) sobre imágenes
   - LeNet sobre MNIST, brevemente
   - Una CNN 2D que hace regresión de tendencias de calentamiento a partir de un campo climático sintético, leída contra modelos de referencia de mínimos cuadrados
   - Un detector de sismos con CNN 1D y su piso de detección, frente a frente con el disparador clásico STA/LTA en las mismas trazas
   - Una prueba de realidad con formas de onda reales de miniPNW: la brecha sintético-real medida, no disimulada con ajustes
   - Leer y recodificar una red publicada

5. **Modelos de secuencias** (4.4)
   - Ventanas de contexto y horizontes de pronóstico
   - RNN simples y por qué los gradientes se desvanecen
   - LSTM, autoatención desde cero y un codificador *transformer* pequeño
   - Todos comparados en la misma tarea de pronóstico, contra modelos de referencia de persistencia e ingenuo estacional

6. **Los tres pilares del desarrollo de modelos** (4.5)
   - Pilar 1: curaduría de los datos de entrenamiento — ruido en las etiquetas, desacuerdo estructurado entre etiquetas, desbalance de clases, ruido de sensores, calidad heterocedástica
   - Pilar 2: arquitectura — ancho, profundidad, modelos de referencia, ensambles profundos y *MC dropout* para la incertidumbre, calibración vs. discriminación, comportamiento fuera de rango
   - Pilar 3: estrategias de entrenamiento — tasa de aprendizaje, tamaño de lote, parada temprana (*early stopping*), planificadores de tasa (*schedulers*)
   - Diagnóstico de entrenamientos rotos a partir de sus curvas de pérdida
   - Búsqueda de hiperparámetros con Optuna

7. ***Autoencoders* (autocodificadores) y autosupervisión** (4.6)
   - *Autoencoders* densos, convolucionales y de eliminación de ruido sobre espectrogramas sísmicos
   - Preentrenamiento con *autoencoder* enmascarado, portado a un segundo dominio (campos climáticos en malla)
   - Reutilizar un codificador preentrenado cuando las etiquetas escasean
   - Una prueba de transferencia con miniPNW: las características cruzan la brecha sintético-real, las fronteras de decisión no

8. **Aprendizaje informado por la física** (4.7)
   - Restricciones físicas como términos de la pérdida
   - Una ablación con la ley de enfriamiento y una PINN de difusión de calor 1D
   - Un modelo de referencia de diferencias finitas que vence a la PINN en el problema directo (~750x más rápido, 100x más exacto)
   - Una PINN inversa que recupera la difusividad a partir de 40 muestras ruidosas, y un ejercicio de romper la PINN por desbalance de la pérdida
   - Dónde están las PINN en 2026, y los operadores neuronales como sucesores

9. **Comparativa de pronóstico de series de tiempo** (4.10)
   - Modelos de referencia, SARIMA, *gradient boosting* (potenciación de gradiente), LSTM y un codificador *transformer* sobre series geocientíficas reales
   - Divisiones temporales honestas y MASE
   - La tabla de clasificación (*leaderboard*) de pronóstico de la clase

10. **Hito del proyecto final** (4.20)
    - Requisitos de exploración de arquitecturas, evaluación y diagnóstico para el hito de aprendizaje profundo

El aprendizaje por transferencia aparece donde se usa: el cuaderno 4.6 cierra con una sonda lineal sobre un codificador preentrenado congelado, que es aprendizaje por transferencia en miniatura. Los modelos de lenguaje de gran escala (LLM) y los agentes de IA se cubren en el capítulo 6.

### Resultados de aprendizaje

Al final de este capítulo, usted podrá:
- Implementar, entrenar y evaluar redes neuronales en PyTorch, desde una sola neurona hasta un codificador *transformer*.
- Diagnosticar problemas de entrenamiento a partir de las curvas de aprendizaje y corregirlos.
- Cuantificar cómo la calidad de los datos, las decisiones de arquitectura y la estrategia de entrenamiento afectan, cada una, el desempeño del modelo.
- Estimar la incertidumbre de las predicciones con ensambles profundos.
- Usar preentrenamiento autosupervisado cuando los datos etiquetados escasean.
- Elegir y comparar modelos de pronóstico con divisiones temporales libres de fugas de datos.

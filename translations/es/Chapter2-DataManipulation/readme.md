# Panorama del capítulo

## Capítulo 2: Datos geocientíficos listos para IA

Este capítulo es el pilar de datos listos para IA del curso. Los proyectos de aprendizaje automático (*machine learning*, **ML**) en las geociencias triunfan o fracasan según la calidad de sus datos, y la mayor parte del trabajo ocurre antes de cualquier modelo: entender qué son los datos, leer y escribir formatos estándar, limpiar tablas, remodelar arreglos, remuestrear, caracterizar distribuciones, transformar y filtrar señales, generar datos sintéticos honestos, construir características (*features*) y reducir la dimensionalidad. El capítulo termina con una lección de cierre que define los datos listos para IA de manera operativa — procedencia, metadatos, tablas ordenadas, particiones de referencia (*benchmark splits*) y controles contra la fuga de datos — y esa definición se califica en el proyecto final. Tres flujos de datos recorren el capítulo y reciben el mismo tratamiento: campos en malla (rásteres climáticos, imágenes), series regulares de sensores de alta tasa (sismogramas a 100 Hz, mareógrafos horarios, GNSS diario) y observaciones puntuales dispersas e irregulares (redes de pozos de agua subterránea, campañas de campo) — porque todo proyecto geocientífico termina manejando al menos dos de ellos.

### El arco del capítulo

Las lecciones se construyen en orden:

1. **2.1 Definiciones de datos** — modalidades de datos en geociencias; arreglos frente a *data frames*; formatos comunes y optimizados para la nube.
2. **2.2 Formatos de datos** — práctica de lectura y escritura de CSV, GeoJSON, GeoTIFF, netCDF, HDF5, Parquet y Zarr; comparación del tamaño en disco.
3. **2.3 DataFrames de Pandas** — series y *data frames*, manejo de fechas y horas, filtrado, agrupamiento, agregación y mapeo de metadatos de estaciones.
4. **2.4 Preparación de DataFrames** — limpieza de una tabla de geoquímica de roca total: datos faltantes, valores centinela, valores censurados en un límite de detección, ausencias informativas, correlaciones y distribuciones por clase.
5. **2.5 Arreglos** — arreglos de NumPy y Xarray, indexado y remodelado, dimensiones etiquetadas, y un primer vistazo a los tensores de PyTorch.
6. **2.6 Remuestreo** — remuestreo estadístico (aleatorización, *bootstrap*, Monte Carlo) para la incertidumbre, y luego remuestreo de señales: diezmado con antisolapamiento de un registro de mareógrafo, interpolación con política de huecos de una serie GNSS degradada, agregación de una red irregular de pozos y el *block bootstrap* para ruido correlacionado — con datos GNSS sintéticos y reales en todo momento.
7. **2.7 Consideraciones estadísticas** — momentos, distribuciones y la ley de Gutenberg-Richter, sobre datos geoquímicos sintéticos y reales.
8. **2.8 Transformadas espectrales** — transformadas de Fourier y de ondículas (*wavelets*) de sismogramas y de campos 2D.
9. **2.9 Filtrado** — filtros pasabajas, pasaaltas y pasabanda; filtros de fase cero frente a causales; separación de tendencia, ciclo estacional y ruido; filtrado a través de huecos y recuperación de un error de reloj en un sismograma real.
10. **2.10 Datos sintéticos** — construcción de sismogramas sintéticos y de ruido con espectro ajustado; una medición resuelta del piso de detección STA/LTA con barras de error binomiales; cuándo los datos sintéticos son admisibles en ciencia.
11. **2.11 Ingeniería de características** — características construidas a mano y automatizadas para series de tiempo, con un *benchmark* real de formas de onda sísmicas.
12. **2.12 Reducción de dimensionalidad** — PCA, EOF sobre campos climáticos, ICA y t-SNE.
13. **2.13 Datos listos para IA (lección de cierre)** — la lista de verificación operativa: fichas de datos, particiones de referencia, la unión ráster-estación para covariables en malla, la fuga por preprocesamiento y las particiones correctas para datos autocorrelacionados.

El capítulo cierra con la **tarea del proyecto final para este pilar (2.20)**: construir un conjunto de datos listo para IA para su propio proyecto, con una ficha de datos y particiones de referencia, siguiendo la lista de verificación de 2.13.

### Resultados de aprendizaje

Al final de este capítulo, usted podrá:

- Reconocer los tipos, modalidades y formatos de datos comunes en las geociencias, incluidos los formatos optimizados para la nube.
- Manipular datos tabulares con Pandas y datos en arreglos con NumPy y Xarray.
- Caracterizar datos con momentos estadísticos, distribuciones y métodos de remuestreo.
- Reparar patologías de instrumentos — huecos, solapamiento espectral (*aliasing*), errores de tiempo, valores censurados — con la reparación calificada contra una verdad de referencia conocida.
- Aplicar transformadas de Fourier y de ondículas y diseñar filtros digitales.
- Generar datos sintéticos con responsabilidad y declarar su uso.
- Construir características y reducir la dimensionalidad para tareas posteriores de ML.
- Ensamblar un conjunto de datos listo para IA con procedencia documentada, una ficha de datos y particiones de referencia a prueba de fugas.

### Tareas

- **Tarea final (2.20)**: Construya un conjunto de datos listo para IA para su proyecto final. Aplique la lista de verificación de 2.13: documente la procedencia y la licencia, escriba una ficha de datos, defina particiones de referencia y demuestre que su preprocesamiento no filtra información entre particiones.

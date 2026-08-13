# Capítulo 6: IA agéntica para la ciencia

En 2023, los modelos de lenguaje autocompletaban código. En 2026, los sistemas agénticos leen repositorios completos, ejecutan código e inspeccionan los resultados, buscan en la literatura y llevan a cabo tareas de investigación de varios pasos con supervisión limitada. Entre esas dos fechas, la habilidad humana útil se desplazó. Ya no es escribir *prompts* (instrucciones) — pedir amablemente, con las palabras correctas, es un problema resuelto. Las habilidades que enseña este capítulo son la **especificación** (enunciar una tarea con la precisión suficiente para que el éxito sea verificable) y la **evaluación** (verificarlo).

Nada de lo que hay aquí contradice los capítulos 1 al 5; depende de ellos. La salida de un agente es un análisis no confiable, y usted ya sabe qué hacer con un análisis no confiable: exigir un entorno reproducible (capítulo 5), una evaluación justa contra una verdad de referencia reservada (capítulo 3) y un modelo de referencia (*baseline*) al que deba superar. Este capítulo aplica esa maquinaria a los agentes mismos.

Una nota sobre cómo trabajamos en este capítulo: los cuadernos no llaman a ningún servicio externo de IA. Corren contra artefactos provistos — respuestas de agentes grabadas, agentes simulados con modos de falla plantados — de modo que se ejecutan en CI y no cuestan nada. El razonamiento se transfiere directamente a los agentes en vivo que usted usa en su proyecto, donde aplica la política de IA del curso: el uso está permitido, declarado y verificado, y usted debe poder defender cada línea que entrega ([capítulo 1.8](../Chapter1-GettingStarted/1.8_ai_in_your_workflow.md)).

## Qué contiene este capítulo

1. **[6.1 De los modelos de lenguaje a los agentes](6.1_llms_to_agents.md)** — Conceptos: qué es y qué no es un LLM basado en *transformer*, por qué los LLM son malas calculadoras, recuperación, uso de herramientas y la alucinación como modo de falla distinto del error ordinario.
2. **[6.2 Evaluación crítica de la salida de una IA](6.2_critical_evaluation.ipynb)** — Práctica: verificar las afirmaciones cuantitativas de un agente contra los datos, detectar una cita fabricada y calificar dos revisiones de IA para ver los sesgos de los jueces en acción — y luego intercambiar calificaciones con un compañero para medir si dos evaluadores que aplican la misma rúbrica siquiera coinciden.
3. **[6.3 Construya un conjunto de evaluación](6.3_build_an_eval_set.ipynb)** — El ejercicio calificado central del capítulo: antes de confiarle una tarea a un agente, construya el conjunto de evaluación (*eval set*) que lo mide. Desarrollado de principio a fin sobre una tarea de velocidad GNSS — especificación, casos, un calificador que sobrevive a salidas malformadas, tolerancias derivadas empíricamente de realizaciones sintéticas — y luego extendido en «Calificar sin verdad computable» a tareas sin respuesta exacta: rúbrica como calificador, acuerdo entre dos evaluadores (porcentaje y kappa de Cohen), controles del sesgo del juez y tasas de éxito sobre repeticiones de un agente estocástico. Una sección final opcional corre el mismo arnés contra un modelo de pesos abiertos en vivo (OLMo 2 vía Ollama); la vía simulada/grabada sin conexión sigue siendo la base calificada. El ejercicio se repite sobre una tarea del dominio de su propio proyecto.
4. **[6.4 Declaración y normas](6.4_disclosure_and_norms.md)** — Atribución y declaración para la investigación asistida por IA: qué esperan las revistas, el formato de declaración del curso, la capa institucional más allá de la revista (política del empleador y del patrocinador, clasificación de datos, retención de transcripciones) y de quién es la corrección (suya).
5. **[6.5 El arco de lectura](6.5_reading_arc.md)** — El contrato culminante del trimestre: una revisión de literatura asistida por IA con un registro de verificación de citas, la anatomía de los buenos artículos, una rúbrica de calidad que usted escribe en el género que declare, y un agente de revisión previa al envío construido a partir de esa rúbrica y evaluado con la maquinaria de este capítulo.

## Resultados de aprendizaje

Al terminar este capítulo usted puede:

- explicar qué es un agente (un LLM, un conjunto de herramientas y un bucle sobre observaciones) y en qué parte de ese bucle entran los errores;
- verificar una afirmación generada por IA contra los datos subyacentes en lugar de contra su impresión de plausibilidad;
- diseñar y correr un conjunto de evaluación con verdad de referencia para una tarea de agente antes de desplegar el agente;
- calificar un agente cuya salida es un juicio y no un número: verificaciones de rúbrica sobre defectos plantados, acuerdo entre evaluadores y tasas de éxito sobre repeticiones;
- declarar la asistencia de IA en el formato que exigen el curso, las políticas vigentes de las revistas y su institución;
- convertir sus propios estándares de calidad en un agente de revisión que usted ha medido, y defender tanto los estándares como la medición.

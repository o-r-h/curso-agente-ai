# Nivel 6 – Agentes autónomos

> **Objetivo**: Crear agentes que sean capaces de **planificar y actuar de forma autónoma**, combinando herramientas, memoria y modelos de lenguaje para resolver problemas de múltiples pasos.

---

## Tabla de contenidos

1. [Introducción a agentes autónomos](#introducción-a-agentes-autónomos)
2. [Arquitecturas de agentes](#arquitecturas-de-agentes)
   - [ReAct](#react)
   - [AutoGPT](#autogpt)
   - [BabyAGI](#babyagi)
3. [Planificación paso a paso (](#planificación-paso-a-paso-planning)[*planning*](#planificación-paso-a-paso-planning)[)](#planificación-paso-a-paso-planning)
4. [Control de bucles y límites](#control-de-bucles-y-límites)
5. [Evaluación y depuración de agentes](#evaluación-y-depuración-de-agentes)
6. [Implementación práctica](#implementación-práctica)
   - [LangGraph](#langgraph)
   - [CrewAI](#crewai)
   - [LangChain Experimental](#langchain-experimental)
7. [Buenas prácticas](#buenas-prácticas)
8. [Checklist de implementación](#checklist-de-implementación)
9. [Glosario](#glosario)
10. [Referencias y librerías recomendadas](#referencias-y-librerías-recomendadas)

---

## Introducción a agentes autónomos

Un **agente autónomo** es un sistema que no solo responde, sino que **decide, planifica y actúa en múltiples pasos** sin intervención humana constante. Estos agentes buscan imitar un *ciclo cognitivo* inspirado en teorías de inteligencia artificial y ciencias cognitivas:

**Percibir → Razonar/Planear → Actuar → Reflexionar → Repetir**

Esto les permite enfrentar tareas complejas que requieren coordinación de varias herramientas, memoria y adaptabilidad.

## Arquitecturas de agentes

### ReAct

- **Idea**: combina **razonamiento textual** con la capacidad de ejecutar **acciones externas**.
- El LLM genera primero un “pensamiento” sobre qué hacer, luego elige una acción.
- Flujo: *Pensamiento → Acción → Observación → Respuesta final*.
- **Uso típico**: agentes que interactúan con bases de conocimiento y APIs.

### AutoGPT

- Arquitectura diseñada para perseguir un **objetivo general**.
- Descompone el objetivo en subtareas y las ejecuta iterativamente.
- Usa memoria a largo plazo para no perder contexto.
- **Ventaja**: alto nivel de autonomía.
- **Desafío**: requiere fuertes mecanismos de control y validación.

### BabyAGI

- Implementación ligera de un agente iterativo.
- Mantiene una lista dinámica de tareas que se **generan, priorizan y ejecutan**.
- Ideal para experimentar con agentes autónomos simples.
- **Tip**: puedes extenderlo integrando bases vectoriales o más herramientas.

## Planificación paso a paso (*planning*)

- Permite a los agentes dividir un **objetivo complejo** en tareas más pequeñas y alcanzables.
- Métodos:
  - **Planificación jerárquica**: divide en subtareas en varios niveles.
  - **LLMs como planificadores**: prompts que piden descomposición de problemas.
  - **Cadena de razonamiento (chain-of-thought)** guiada.

**Ejemplo práctico**:

```text
Objetivo: Planear un viaje a Madrid
1. Buscar vuelos
2. Reservar hotel
3. Crear itinerario
4. Calcular presupuesto
5. Confirmar reservas
```

**Tip**: simula escenarios con tareas sencillas antes de pasar a flujos más complejos.

## Control de bucles y límites

Los agentes autónomos pueden caer en **loops infinitos** o ejecutar tareas irrelevantes.

- **Límites de iteraciones**: establecer un máximo de pasos por ejecución.
- **Restricciones de tiempo**: cortar procesos que exceden cierto tiempo.
- **Watchdog**: proceso externo que monitorea al agente.
- **Criterios de confianza**: detenerse si no se cumple un umbral de certeza.
- **Tip**: combina límites duros (máximo 10 pasos) con límites suaves (detener si se repite información).

## Evaluación y depuración de agentes

- **Logging detallado**: guarda los pensamientos, acciones y observaciones.
- **Replay**: reproducir ejecuciones pasadas para depurar.
- **Métricas clave**:
  - % de objetivos completados.
  - Latencia promedio.
  - Errores de llamadas a herramientas.
  - Tasa de bucles detectados.
- **Tip**: utiliza datasets sintéticos de prueba antes de poner un agente en producción.

## Implementación práctica

### LangGraph

- Permite definir agentes como **grafos de estados**.
- Útil para controlar el flujo entre razonamiento, acciones y memoria.
- Ejemplo: un nodo “buscar info” → nodo “resumir” → nodo “responder”.

### CrewAI

- Orientado a **agentes colaborativos**.
- Permite asignar sub-agentes especializados (ej. “Agente investigador”, “Agente redactor”).
- Útil en flujos donde se requiere especialización y colaboración.

### LangChain Experimental

- Ideal para prototipar agentes.
- Incluye **agentes de reflexión**, que analizan sus propias salidas antes de responder.
- **Tip**: perfecto para experimentar con “metacognición” del agente.

## Buenas prácticas

- Define **objetivos claros y verificables**.
- Empieza con **entornos simulados** y escala poco a poco.
- Controla recursos: memoria, tokens, llamadas a API.
- Implementa siempre un mecanismo de **detención segura**.
- Documenta la lógica de planificación y decisiones.
- Usa métricas de desempeño para iterar y mejorar.
- **Tip**: integra feedback humano (*human-in-the-loop*) en primeras versiones.

## Checklist de implementación

-

## Glosario

- **ReAct**: arquitectura que combina razonamiento y acción.
- **AutoGPT**: agente autónomo orientado a objetivos.
- **BabyAGI**: agente que genera y ejecuta tareas en un bucle.
- **Planning**: proceso de descomponer un objetivo en pasos.
- **Watchdog**: mecanismo de control para detener bucles.

## Referencias y librerías recomendadas

- **langgraph**: agentes multi‑paso basados en grafos.
- **crewai**: agentes autónomos colaborativos.
- **langchain‑experimental**: prototipos de agentes.
- **LangChain**: marco general de orquestación de LLMs y agentes.


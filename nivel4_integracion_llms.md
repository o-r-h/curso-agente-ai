# Nivel 4 – Integración con modelos de lenguaje

> **Objetivo**: Usar LLMs (modelos de lenguaje grandes) para **responder** y **tomar decisiones** de forma fiable, integrándolos con memoria, herramientas y recuperación de conocimiento (RAG), controlando el formato de salida y mitigando alucinaciones.

---

## Tabla de contenidos
1. [Panorama general y arquitectura](#panorama-general-y-arquitectura)
2. [Modelos locales vs APIs (OpenAI, Hugging Face)](#modelos-locales-vs-apis-openai-hugging-face)
3. [Prompts y *prompt engineering*](#prompts-y-prompt-engineering)
   - [Estructura jerárquica (sistema/desarrollador/usuario)](#estructura-jerárquica-sistemadesarrolladorusuario)
   - [Plantillas, *few-shot* y roles](#plantillas-few-shot-y-roles)
   - [Instrucciones de seguridad y políticas](#instrucciones-de-seguridad-y-políticas)
4. [Control de salida y reducción de alucinaciones](#control-de-salida-y-reducción-de-alucinaciones)
   - [Parámetros de decodificación](#parámetros-de-decodificación)
   - [Salidas estructuradas (JSON Schema)](#salidas-estructuradas-json-schema)
   - [Herramientas / *function calling* / *tool use*](#herramientas--function-calling--tool-use)
   - [Verificación, *self-checks* y *guardrails*](#verificación-self-checks-y-guardrails)
5. [Sistemas RAG (*Retrieval-Augmented Generation*)](#sistemas-rag-retrieval-augmented-generation)
   - [Ingesta y chunking](#ingesta-y-chunking)
   - [Índice vectorial y recuperación](#índice-vectorial-y-recuperación)
   - [Plantillas RAG y citación](#plantillas-rag-y-citación)
   - [Evaluación de RAG](#evaluación-de-rag)
6. [Práctica con código](#práctica-con-código)
   - [API (OpenAI): chat, JSON, herramientas](#api-openai-chat-json-herramientas)
   - [Local (Transformers): *pipeline* y *generate()*](#local-transformers-pipeline-y-generate)
   - [RAG mínimo con LangChain + Chroma](#rag-mínimo-con-langchain--chroma)
7. [Observabilidad, versiones y costos](#observabilidad-versiones-y-costos)
8. [Buenas prácticas y anti‑patrones](#buenas-prácticas-y-anti-patrones)
9. [Checklist de producción](#checklist-de-producción)
10. [Glosario](#glosario)
11. [Librerías recomendadas](#librerías-recomendadas)

---

## Panorama general y arquitectura
**Integrar LLMs** implica orquestar: (a) **prompts** bien diseñados, (b) **memoria** (corto/largo plazo), (c) **herramientas** (APIs, DBs), (d) **recuperación** (RAG) y (e) **controles** de seguridad/formatos.

```
Usuario ↔ Orquestador/Agente
        ├─ LLM (API o local)
        ├─ Memoria (buffer, resúmenes, entidades)
        ├─ RAG (vector store + filtros + re‑ranking)
        ├─ Herramientas/Funciones (APIs, DBs, ejecutores)
        └─ Validación de salida (JSON Schema / gramáticas / reglas)
                         ↑ Observabilidad (trazas, costos, latencia)
```

## Modelos locales vs APIs (OpenAI, Hugging Face)
**APIs** (p. ej., OpenAI):
- Pros: *SOTA*, fácil de usar, *uptime* administrado, características avanzadas (salida estructurada, herramientas, razonamiento).
- Contras: dependencia externa, costos por token, requisitos de cumplimiento/PII.

**Locales** (p. ej., LLaMA/Mistral con Transformers/TGI/Ollama):
- Pros: control total, datos internos sin salir, optimización fina (cuantización, *throughput*), costos predecibles en cargas altas.
- Contras: *DevOps/ML Ops* (servir, escalar, GPUs), actualización de modelos, *guardrails* a cargo del equipo.

**Criterios de elección**: privacidad, latencia, *SLA*, presupuesto, *tooling* disponible (funciones/JSON/correcciones), *evals* internas, compatibilidad con memoria y RAG.

## Prompts y *prompt engineering*
### Estructura jerárquica (sistema/desarrollador/usuario)
- **Sistema**: reglas inquebrantables (rol, tono, límites, seguridad).
- **Desarrollador**: *chain-of-thought guardado*, planes de acción y formato esperado (no expongas razonamiento interno al usuario).
- **Usuario**: consulta/pedidos concretos.

### Plantillas, *few-shot* y roles
- Diseña **plantillas parametrizadas** con *placeholders* (p. ej., `{context}`, `{instrucciones}`, `{formato_json}`).
- Usa **few‑shot** con ejemplos reales y **contrajemplos** (qué *no* hacer). 
- Añade **criterios de calidad** (brevedad, citas, justificación ligera cuando aplique) y **criterios de rechazo** (cuando no haya contexto suficiente).

**Ejemplo de prompt (RAG + JSON)**
```
[Sistema]
Eres un asistente experto. Si el contexto no cubre la pregunta, di "No hay evidencia suficiente en el contexto".

[Desarrollador]
Responde en JSON con las claves {"respuesta", "fuentes"}. No inventes citas.

[Usuario]
Pregunta: {pregunta}

[Contexto recuperado]
{contexto}
```

### Instrucciones de seguridad y políticas
- Define explícitamente: PII, contenido inseguro, límites de acciones, *rate limits*, *timeouts* y *fallbacks* (p. ej., transferir a humano).
- Añade *guardrails* programáticos además de instrucciones (validación JSON, filtros, *allowlists*).

## Control de salida y reducción de alucinaciones
### Parámetros de decodificación
- **`temperature`** (aleatoriedad) y **`top_p`** (nucleus) → menor = más determinista (útil para flujos operativos/JSON).
- **`max_tokens`**, **`stop`** y penalizaciones (frecuencia/presencia) para estilo/repetición.

### Salidas estructuradas (JSON Schema)
- Obliga al LLM a **respetar un esquema** (campos requeridos, tipos, *enums*). Reduce *parsing errors* y ambigüedad.
- Alternativas locales: **gramáticas (regex/JSON)** con servidores como TGI; validación post‑hoc con **Pydantic**.

### Herramientas / *function calling* / *tool use*
- Define funciones con **nombre, descripción y parámetros**. El LLM decide **cuándo** llamarlas y con **qué argumentos**.
- Patrón de bucle: *mensaje → decisión → llamada herramienta → inyectar resultado → respuesta final*.

### Verificación, *self-checks* y *guardrails*
- *Self‑check* del modelo (p. ej., "¿está sustentado por el contexto?").
- *Double pass* (respuesta → crítica → versión final). 
- Umbral de similitud y **no responder** si el contexto es insuficiente.
- *Post‑validation*: JSON Schema, reglas de negocio, *allowlist/denylist*, unit tests de prompts.

## Sistemas RAG (*Retrieval-Augmented Generation*)
### Ingesta y chunking
- Limpia HTML/boilerplate; segmenta en **200–400 tokens** con **solapamiento**.
- Añade metadatos: `fuente`, `fecha`, `autoridad`, `user_id`, `idioma`.

### Índice vectorial y recuperación
- Embeddings normalizados + similitud coseno / *inner product*.
- Estrategias: **MMR**, filtros por metadatos, búsqueda **híbrida** (BM25 + densa), re‑ranking por *cross‑encoder*.

### Plantillas RAG y citación
- Pide **citas** `{[fuente, título, fecha]}` y un **grado de confianza**.
- Incluye instrucción de **rechazo** si el contexto no respalda la respuesta.

### Evaluación de RAG
- Métricas de recuperación: *Recall@k*, *nDCG*, latencia.
- Métricas E2E: *answer faithfulness/groundedness*, relevancia, utilidad.
- *A/B* con y sin RAG; *evals* sintéticas + revisión humana.

## Práctica con código
### API (OpenAI): chat, JSON, herramientas
> Ejemplo orientativo: chat con **salida JSON** + **herramienta** de clima.

```python
# pip install openai
from openai import OpenAI
client = OpenAI()

schema = {
  "name": "RespuestaClima",
  "schema": {
    "type": "object",
    "properties": {
      "respuesta": {"type": "string"},
      "ciudad": {"type": "string"},
      "fuentes": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["respuesta", "ciudad"],
    "additionalProperties": False
  }
}

# Definición de herramienta (function calling)
tools = [{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Devuelve clima actual para una ciudad",
    "parameters": {
      "type": "object",
      "properties": {"city": {"type": "string"}},
      "required": ["city"]
    }
  }
}]

messages = [
  {"role": "system", "content": "Responde solo con JSON válido conforme al esquema."},
  {"role": "user", "content": "¿Necesitaré paraguas hoy en Monterrey?"}
]

# Respuesta con salida estructurada (Responses API)
resp = client.responses.create(
  model="gpt-4.1-mini",
  input=messages,
  tools=tools,
  response_format={
    "type": "json_schema",
    "json_schema": schema
  }
)

# Si el modelo decide usar la herramienta, procesa el tool call y re‑inyecta el resultado
# (pseudocódigo) → resp.output_tool_calls -> ejecutar -> messages.append(tool_result) -> nueva llamada
print(resp.output_text)  # JSON conforme al schema
```

> **Notas**: usa *retries* exponenciales, *timeouts*, *streaming* si necesitas latencia percibida baja, y registra `prompt/version`.

### Local (Transformers): *pipeline* y *generate()*
> Cargar un modelo instruct (Mistral/LLaMA), controlar decodificación y formatear diálogo.

```python
# pip install transformers accelerate torch --upgrade
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

chat = pipeline(
    "text-generation",
    model=model,
    tokenizer=tok,
    device_map="auto"
)

prompt = (
    "[INST] Sistema: Responde en JSON con claves {respuesta, fuentes}. "
    "Usuario: ¿Cómo está el clima en Monterrey? [/INST]"
)

out = chat(
    prompt,
    max_new_tokens=256,
    do_sample=False,      # determinista
    temperature=0.2,
    top_p=0.9,
    eos_token_id=tok.eos_token_id,
    pad_token_id=tok.eos_token_id
)[0]["generated_text"]
print(out)
```

> Para **salida estricta JSON** en local, considera servir con **TGI** (JSON/regex grammars) o validar con **Pydantic** y re‑intentar.

### RAG mínimo con LangChain + Chroma
> Integra recuperación semántica con una LLM (API o local) y pide **citas**.

```python
# pip install langchain chromadb sentence-transformers
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 1) Ingesta
raw_docs = [("POLÍTICA", "Los cambios de vuelo permiten una modificación sin costo en tarifa flexible."),
            ("GUÍA", "Monterrey tiene temporada de lluvias en septiembre.")]

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
docs = []
for title, text in raw_docs:
    for i, ch in enumerate(splitter.split_text(text)):
        docs.append({"page_content": ch, "metadata": {"title": title, "i": i}})

# 2) Índice vectorial
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", encode_kwargs={"normalize_embeddings": True})
vs = Chroma.from_documents([type("Doc", (), d)() for d in docs], embedding=emb, persist_directory="./chroma_llm")
retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 12})

# 3) Prompt + LLM
TEMPLATE = (
    "Eres un asistente. Usa SOLO el contexto. Si no hay evidencia suficiente, responde 'No hay evidencia suficiente'.\n"
    "Devuelve JSON: {respuesta, fuentes}.\n\n"
    "Pregunta: {q}\n\nContexto:\n{context}\n"
)
prompt = ChatPromptTemplate.from_template(TEMPLATE)
llm = ChatOpenAI(temperature=0)

# 4) Consulta
q = "¿Puedo cambiar de vuelo sin pagar?"
ctx_docs = retriever.get_relevant_documents(q)
context = "\n".join(f"- [{d.metadata['title']}] {d.page_content}" for d in ctx_docs)

msg = prompt.format_messages(q=q, context=context)
ans = llm.invoke(msg)
print(ans.content)
```

## Observabilidad, versiones y costos
- **Versiona prompts** (IDs y *changelogs*). 
- Registra **tokens, latencia, *tools* invocadas y *context size***; traza *errors* y reintentos.
- **Caching** de respuestas/embeddings; re‑uso de contexto.
- Monitoriza **groundedness** en RAG y *drift* de fuentes.

## Buenas prácticas y anti‑patrones
**Buenas prácticas**
- Empieza con un **baseline** determinista (baja `temperature`) y JSON Schema.
- Aplica **RAG** para hechos y **tools** para acciones/consultas.
- Establece **políticas de rechazo** y *fallbacks* a humano.
- Usa **tests automatizados** de prompts y contratos de salida (JSON/gramática).
- Separa **conocimiento mutable** (RAG) del **comportamiento** (prompt del sistema).

**Anti‑patrones**
- Plantillas no versionadas o editadas en caliente sin *rollbacks*.
- Inyectar contexto **irrelevante** o excesivo (diluye la señal).
- Confiar en *temperature* para arreglar alucinaciones (usa grounding/herramientas en su lugar).
- No validar salidas antes de llamar APIs críticas.

## Checklist de producción
- [ ] Prompts versionados y *changelogs*.
- [ ] JSON Schema/gramática + validación y re‑intentos.
- [ ] RAG con métricas de recuperación y *guardrails* de citación.
- [ ] Tool calling con *timeouts*, *circuit breakers*, *retries*.
- [ ] Telemetría de tokens, costo, latencia, errores.
- [ ] Playbooks de caídas y *rate limiting*.
- [ ] Revisión de seguridad/PII y cumplimiento.

## Glosario
- **LLM**: modelo de lenguaje grande.
- **Tool/Function Calling**: mecanismo para que el LLM invoque herramientas externas.
- **JSON Schema**: contrato de salida para validar estructura y tipos.
- **RAG**: generación aumentada por recuperación de contexto.
- **MMR**: selección que balancea relevancia y diversidad.

## Librerías recomendadas
- **openai** o **transformers** → conexión a LLMs (API o local). 
- **langchain** → orquestación, prompts y conexión con memoria/RAG. 
- **transformers** → modelos locales (LLaMA, Mistral) y control de decodificación.

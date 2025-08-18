# Nivel 3 – Memoria en agentes

> **Objetivo**: Diseñar y construir memoria en agentes conversacionales para que recuerden el contexto y lo usen de forma segura y eficaz durante la conversación y a lo largo del tiempo.

---

## Tabla de contenidos
1. [Qué entendemos por “memoria” en un agente](#qué-entendemos-por-memoria-en-un-agente)
2. [Tipos de memoria](#tipos-de-memoria)
   - [Corto plazo (contexto inmediato)](#corto-plazo-contexto-inmediato)
   - [Largo plazo (histórico)](#largo-plazo-histórico)
   - [Variantes útiles: resumen, entidades, episodios](#variantes-útiles-resumen-entidades-episodios)
3. [Representación del contexto en vectores](#representación-del-contexto-en-vectores)
   - [Embeddings y elección de modelo](#embeddings-y-elección-de-modelo)
   - [Chunking, ventanas deslizantes y metadatos](#chunking-ventanas-deslizantes-y-metadatos)
   - [Métricas de similitud y normalización](#métricas-de-similitud-y-normalización)
4. [Bases de datos vectoriales (Vector Stores)](#bases-de-datos-vectoriales-vector-stores)
   - [FAISS](#faiss)
   - [ChromaDB](#chromadb)
   - [Persistencia auxiliar con SQLite](#persistencia-auxiliar-con-sqlite)
5. [Recuperación de contexto relevante (Retrieval)](#recuperación-de-contexto-relevante-retrieval)
   - [Búsqueda densa y ANN](#búsqueda-densa-y-ann)
   - [Re-ranking, MMR e híbridos](#re-ranking-mmr-e-híbridos)
   - [Recuperación conversacional](#recuperación-conversacional)
6. [Arquitectura de un agente con memoria](#arquitectura-de-un-agente-con-memoria)
7. [Práctica con código](#práctica-con-código)
   - [Instalación](#instalación)
   - [Memoria de corto plazo con LangChain](#memoria-de-corto-plazo-con-langchain)
   - [Memoria de largo plazo con FAISS + Sentence-Transformers](#memoria-de-largo-plazo-con-faiss--sentence-transformers)
   - [Conversational RAG con ChromaDB + LangChain](#conversational-rag-con-chromadb--langchain)
   - [Persistencia de metadatos en SQLite](#persistencia-de-metadatos-en-sqlite)
8. [Evaluación](#evaluación)
9. [Buenas prácticas y anti-patrones](#buenas-prácticas-y-anti-patrones)
10. [Checklist de producción](#checklist-de-producción)
11. [Glosario](#glosario)
12. [Librerías recomendadas](#librerías-recomendadas)

---

## Qué entendemos por “memoria” en un agente
**Memoria** es cualquier mecanismo que conserva información útil más allá de un turno de diálogo para mejorar comprensión, planeación y acciones. En la práctica, abarca: buffers de contexto, resúmenes, extracción de entidades clave, y almacenes vectoriales indexados para búsqueda semántica.

**Objetivos principales**:
- Reducir repreguntas (menos fricción).
- Aumentar precisión (usar lo ya dicho o sabido).
- Mantener coherencia (preferencias, restricciones, historial).
- Cumplir requisitos de seguridad/privacidad (qué recordar, por cuánto tiempo y cómo).

## Tipos de memoria
### Corto plazo (contexto inmediato)
- **Buffer de turnos**: últimas *N* interacciones tal cual (texto bruto). 
- **Ventana resumida**: resumen incrementa la “capacidad efectiva” cuando la ventana de tokens es limitada.

**Cuándo usar**: seguimiento de una tarea en curso, resolución de anáforas (*“esa” reserva*), desambiguación inmediata.

### Largo plazo (histórico)
- **Vector store**: fragmentos de conocimiento y episodios relevantes embebidos (embeddings) y buscados por similitud.
- **Preferencias del usuario**: p. ej., aerolíneas favoritas, políticas; normalmente estructurado y con TTL.

**Cuándo usar**: recordar hechos pasados que exceden la ventana; reutilizar conocimiento de sesiones previas.

### Variantes útiles: resumen, entidades, episodios
- **Resumen**: condensar el buffer a hitos/decisiones.
- **Memoria de entidades**: tabla de *slots* (persona, empresa, fechas), actualizable y consultable.
- **Episódica**: eventos con tiempo/lugar ("el 3/05 agendamos demo").
- **Semántica**: conocimiento general (FAQs, políticas internas).

## Representación del contexto en vectores
### Embeddings y elección de modelo
- **Sentence-level** (ej. *all-MiniLM-L6-v2*): rápidos (384 dims), buen rendimiento general.
- **Domain-specific**: si tu dominio es técnico/legal, considera modelos ajustados.
- **Normalización**: para coseno, conviene normalizar a norma 1.

### Chunking, ventanas deslizantes y metadatos
- **Tamaño de chunk**: 200–400 tokens suele equilibrar granularidad y contexto; usa **solapamiento** (p. ej., 50 tokens) para continuidad.
- **Estrategias**: por párrafo, por títulos, o **segmentación semántica** (detectar límites temáticos).
- **Metadatos**: `user_id`, `ts`, `fuente`, `tipo`, `título`, `intención` → habilitan filtros y recency.

### Métricas de similitud y normalización
- **Cosine similarity** (común en sent-embeddings). 
- **Dot-product / inner product** (si el modelo lo asume). 
- **L2** (menos común con embeddings normalizados). 

## Bases de datos vectoriales (Vector Stores)
### FAISS
- Biblioteca C++/Python para **búsqueda aproximada**.
- Índices típicos: `IndexFlatIP` (exacto), `IVF` (inverted lists), `HNSW` (grafo pequeño mundo), `PQ/OPQ` (cuantización para memoria).
- Pros: rendimiento y flexibilidad; Contras: requiere gestionar metadatos/persistencia fuera del índice.

### ChromaDB
- Vector DB embebida en Python con **persistencia** y **metadatos** integrados.
- Útil para prototipado rápido y *local-first*.

### Persistencia auxiliar con SQLite
- Guarda **metadatos** y relaciones (documento↔chunk, permisos, TTL). 
- Puede convivir con FAISS/Chroma (ID externo = clave en SQLite).

## Recuperación de contexto relevante (Retrieval)
### Búsqueda densa y ANN
1. **Embeddear** la consulta.
2. **ANN search** (FAISS/Chroma) → top-*k* candidatos.
3. **Filtrado por metadatos** (usuario, fecha, fuente).

### Re-ranking, MMR e híbridos
- **MMR (Maximal Marginal Relevance)**: balancea relevancia y diversidad (evita duplicados semánticos).
- **Híbrida**: BM25 (lexical) + vectorial → robusta a *out-of-vocabulary* y números/códigos.
- **Re-ranking con cross-encoders**: mayor precisión (más costo/latencia).

### Recuperación conversacional
- **Condensación de consulta**: reescribir la pregunta actual con historial.
- **Memoria + conocimiento**: combinar episodios del usuario y base de conocimiento.
- **Política**: límites de *k*, umbral de similitud, prioridad por recencia/autoridad.

## Arquitectura de un agente con memoria
```
Usuario ↔ Orquestador
        ├─ Memoria corto plazo (buffer/resumen/entidades)
        ├─ Memoria largo plazo (vector store + metadatos)
        ├─ Recuperador (k, filtros, MMR, re‑ranking)
        ├─ Herramientas/APIs (fuentes de verdad)
        └─ NLG (generación de respuesta) + Política de escritura a memoria
                         ↑                 ↓
                 Evaluación/telemetría   Persistencia/TTL/privacidad
```

**Política de escritura**: qué guardar (decisiones, preferencias, identificadores); cuándo resumir; cuándo olvidar (TTL); y quién puede leerlo (scoping por `user_id`/`org_id`).

## Práctica con código
### Instalación
```bash
pip install langchain sentence-transformers faiss-cpu chromadb sqlite-utils
```

> Si usas GPU, reemplaza `faiss-cpu` por `faiss-gpu`.

### Memoria de corto plazo con LangChain
**Buffer de turnos** y **resumen**:
```python
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
# Para el resumen necesitas un LLM. Ej.:
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(temperature=0)

buffer_memory = ConversationBufferWindowMemory(k=6, return_messages=True)
summary_memory = ConversationSummaryMemory(llm=llm)

# Simular diálogo
buffer_memory.save_context({"human": "Quiero vuelos a Madrid"}, {"ai": "¿Desde qué ciudad y fechas?"})
buffer_memory.save_context({"human": "Desde CDMX del 12 al 18 de octubre"}, {"ai": "Perfecto, ¿preferencia de aerolínea?"})

# Obtener contexto reciente
contexto_corto = buffer_memory.load_memory_variables({})["history"]
# Actualizar resumen cuando crezca el historial
summary_memory.save_context({"input": "Confirmo Aeroméxico"}, {"output": "Buscando opciones"})
resumen = summary_memory.load_memory_variables({})["history"]
```

**Memoria de entidades** (slots clave) con un dict controlado por política:
```python
entidades = {"origen": "CDMX", "destino": "Madrid", "fechas": ("2025-10-12","2025-10-18"), "aerolinea": "Aeroméxico"}
# Usa validadores (pydantic) y TTL por campo si aplica.
```

### Memoria de largo plazo con FAISS + Sentence-Transformers
**Indexar conversaciones/documentos** y consultar por similitud:
```python
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

modelo = SentenceTransformer('all-MiniLM-L6-v2')
# Ejemplo de "episodios" (mensajes/decisiones históricas)
chunks = [
  {"id": 1, "text": "El usuario prefiere pasillo y equipaje documentado", "user_id": "u123", "ts": 1693000000},
  {"id": 2, "text": "Reservamos hotel en Barcelona el 5 de mayo", "user_id": "u123", "ts": 1714900000},
]

# Embeddings normalizados para coseno
X = modelo.encode([c["text"] for c in chunks], normalize_embeddings=True)
index = faiss.IndexFlatIP(X.shape[1])  # inner product = coseno si normalizado
index.add(np.array(X, dtype='float32'))

# Mapa de metadatos (gestión fuera de FAISS)
meta = {i: chunks[i] for i in range(len(chunks))}

# Consulta
q = "¿Cuál es la preferencia de asiento del usuario?"
qv = modelo.encode([q], normalize_embeddings=True).astype('float32')
scores, ids = index.search(qv, k=3)
resultados = [{"score": float(scores[0][i]), **meta[int(ids[0][i])]} for i in range(len(ids[0])) if int(ids[0][i]) != -1]
```

**MMR simple** (diversidad) tras top‑k:
```python
def mmr(query_vec, cand_vecs, lamb=0.5, k=3):
    import numpy as np
    selected, remaining = [], list(range(len(cand_vecs)))
    sims = cand_vecs @ query_vec.T  # (n,1)
    while remaining and len(selected) < k:
        if not selected:
            i = int(np.argmax(sims[remaining]))
            selected.append(remaining.pop(i))
        else:
            max_div = -1; best=-1
            for j, ridx in enumerate(remaining):
                diversity = np.max(cand_vecs[selected] @ cand_vecs[ridx]) if selected else 0
                score = lamb * sims[ridx] - (1-lamb) * diversity
                if score > max_div:
                    max_div, best = score, j
            selected.append(remaining.pop(best))
    return selected
```

### Conversational RAG con ChromaDB + LangChain
```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# 1) Ingesta y chunking
docs = [
  ("POLÍTICAS", "Los cambios de vuelo permiten una modificación sin costo en tarifa flexible."),
  ("PREFERENCIAS", "El usuario prefiere asientos de pasillo y equipaje documentado."),
]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
chunks = []
for title, text in docs:
    for i, ch in enumerate(text_splitter.split_text(text)):
        chunks.append({"page_content": ch, "metadata": {"title": title, "i": i}})

# 2) Embeddings y vector store
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", encode_kwargs={"normalize_embeddings": True})
vs = Chroma.from_documents(
    documents=[type("Doc", (), c)() for c in chunks],
    embedding=emb,
    persist_directory="./chroma_mem"
)

# 3) Retriever con MMR y filtros
retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 12})

# 4) Cadena conversacional con memoria corto plazo
llm = ChatOpenAI(temperature=0)
conv_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever, memory=conv_memory)

respuesta = qa({"question": "¿Puedo cambiar de vuelo sin costo?"})
print(respuesta["answer"])  # usará contexto recuperado + historial
```

### Persistencia de metadatos en SQLite
```python
import sqlite3
conn = sqlite3.connect("memoria.db")
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS episodios (
  id INTEGER PRIMARY KEY,
  user_id TEXT,
  ts INTEGER,
  texto TEXT,
  etiqueta TEXT
);
""")
cur.executemany(
  "INSERT INTO episodios(user_id, ts, texto, etiqueta) VALUES (?,?,?,?)",
  [("u123", 1714900000, "Reservamos hotel en Barcelona el 5 de mayo", "booking")]
)
conn.commit()
```

> **Nota**: guarda en SQLite metadatos y permisos; guarda los vectores en FAISS/Chroma y usa un campo `vector_id` para cruzar.

## Evaluación
**Recuperación**
- *Recall@k* de recuerdos relevantes etiquetados.
- *MRR / nDCG* si tienes orden de relevancia.
- Latencia P50/P95 y % de consultas sin contexto (por debajo de umbral).

**E2E conversacional**
- Tasa de resolución, reducción de repreguntas, coherencia evaluada con rúbrica.
- A/B con memoria vs sin memoria.

**Calidad de memoria**
- Precisión de entidades recordadas; tasa de obsolescencia (recuerdos caducos que afectan la respuesta).

## Buenas prácticas y anti-patrones
**Buenas prácticas**
- Define **política de memoria**: qué guardar, por cuánto tiempo (TTL), quién accede; registra consentimiento del usuario.
- Chunking conservador (200–400 tokens) + **solapamiento**; añade metadatos ricos.
- Normaliza embeddings y usa **MMR**; considera búsqueda híbrida para datos con alfanuméricos/códigos.
- **Umbral de similitud** + *k* pequeño (3–6); si nada supera el umbral, no inyectes contexto.
- Escribe **resúmenes** periódicos para mantener el buffer útil.
- Versiona índices; mantén proceso de **re-ingesta** cuando cambie el modelo de embeddings.

**Anti-patrones**
- Indexar texto ruidoso sin limpiar (URLs largas, boilerplate, banners).
- Chunks enormes (pierdes precisión) o minúsculos (pierdes contexto).
- Confiar solo en vectorial: añade lexical o re-ranking si hay confusiones.
- Inyectar recuerdos con similitud baja (alucinaciones inducidas por contexto).
- No gestionar **PII** (enmascarar, eliminar bajo pedido, cifrar en reposo/transporte).

## Checklist de producción
- [ ] Política de memoria (scoping, TTL, consentimiento, borrado/"derecho al olvido").
- [ ] Telemetría de retrieval (recall@k, latencia, umbrales) y negocio.
- [ ] Mecanismo de resumen y compactación del buffer.
- [ ] Estrategia de re-indexado al cambiar embeddings.
- [ ] Backups y pruebas de restauración del vector store.
- [ ] Guardrails: límite de *k*, umbral, filtros de metadatos, PII.

## Glosario
- **Embeddings**: vectores densos que capturan similitud semántica.
- **Vector store**: base de datos optimizada para búsqueda por similitud en alta dimensión.
- **MMR**: criterio que penaliza redundancia al seleccionar resultados.
- **RAG**: *Retrieval-Augmented Generation*, generación guiada por documentos recuperados.
- **TTL**: tiempo de vida de un registro en memoria.

## Librerías recomendadas
- **langchain** → manejo de memoria conversacional y cadenas de recuperación.
- **faiss** o **chromadb** → almacenamiento y búsqueda vectorial.
- **sqlite3** (opcional) → persistencia de metadatos simple y portable.

# Nivel 7 – Escalabilidad y despliegue

> **Objetivo**: Llevar tu agente a **producción**, exponiéndolo por API, optimizando costo/latencia, observándolo en tiempo real y conectándolo con apps web/móviles. Este nivel combina ingeniería de software, MLOps y prácticas de fiabilidad.

---

## Tabla de contenidos

1. [Arquitectura de referencia](#arquitectura-de-referencia)
2. [APIs REST para exponer el agente](#apis-rest-para-exponer-el-agente)
   - [Diseño de contratos](#diseño-de-contratos)
   - [Streaming (SSE/WebSocket)](#streaming-ssewebsocket)
   - [Versionado, CORS y seguridad](#versionado-cors-y-seguridad)
3. [Optimización de modelos para menor costo/latencia](#optimización-de-modelos-para-menor-costolatencia)
   - [Optimización de prompts y respuestas](#optimización-de-prompts-y-respuestas)
   - [Caching y batching](#caching-y-batching)
   - [Servidores de inferencia y cuantización](#servidores-de-inferencia-y-cuantización)
   - [RAG eficiente y vector stores](#rag-eficiente-y-vector-stores)
4. [Monitoreo y logging](#monitoreo-y-logging)
   - [Logs estructurados y trazas](#logs-estructurados-y-trazas)
   - [Métricas de negocio y técnicas](#métricas-de-negocio-y-técnicas)
   - [Alertas y SLOs](#alertas-y-slos)
5. [Integración con aplicaciones web o móviles](#integración-con-aplicaciones-web-o-móviles)
   - [Frontends web (Next.js/React)](#frontends-web-nextjsreact)
   - [Móvil (React Native/Flutter)](#móvil-react-nativeflutter)
   - [Autenticación y rate limiting](#autenticación-y-rate-limiting)
6. [Despliegue y operación](#despliegue-y-operación)
   - [Dockerfile y contenedorización](#dockerfile-y-contenedorización)
   - [Ejecución y escalado](#ejecución-y-escalado)
   - [Entornos, secretos y migraciones](#entornos-secretos-y-migraciones)
7. [Buenas prácticas y anti‑patrones](#buenas-prácticas-y-anti-patrones)
8. [Checklist de producción](#checklist-de-producción)
9. [Librerías recomendadas](#librerías-recomendadas)
10. [Apéndices de código](#apéndices-de-código)

---

## Arquitectura de referencia

```
Cliente (Web/Móvil)
   │   HTTPS (REST/SSE/WebSocket)
   ▼
API Gateway / Ingress (Auth, CORS, Rate Limit)
   ▼
Servicio de Agente (FastAPI/Flask + Uvicorn)
   ├─ Orquestador (prompts, herramientas, políticas)
   ├─ Memoria (buffer/resúmenes/entidades)
   ├─ Recuperación (RAG: vector store + filtros)
   ├─ Conectores (APIs: pagos, clima, CRM, DB)
   └─ Cliente LLM (API o servidor local)

Infra compartida
   ├─ Vector Store (Chroma/FAISS/Weaviate/PGVector)
   ├─ Cache (Redis/Memcached)
   ├─ Observabilidad (Prometheus + Grafana / OpenTelemetry)
   └─ Cola de trabajos (Celery/RQ) para tareas pesadas
```

> **Tip**: separa **control plane** (orquestación del agente) del **data plane** (servicios de inferencia/embeddings) para escalar de forma independiente.

---

## APIs REST para exponer el agente

### Diseño de contratos

- **Modelos (Pydantic)**: define `Request`/`Response` claras: entrada del usuario, opciones (idioma, temperatura), y salida (texto, citas, tool calls).
- **Errores**: usa códigos HTTP y un esquema de error estándar (`code`, `message`, `details`).
- **Idempotencia**: `request_id` para reintentos seguros.
- **Versionado**: prefija rutas (`/v1/…`).

**Ejemplo (FastAPI) – chat síncrono**

```python
# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI(title="Agent API", version="1.0.0")

class ToolCall(BaseModel):
    name: str
    arguments: dict

class ChatRequest(BaseModel):
    user_id: str
    message: str
    temperature: float = 0.2
    request_id: Optional[str] = None

class ChatResponse(BaseModel):
    text: str
    tool_calls: List[ToolCall] = []
    sources: List[str] = []

@app.post("/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # 1) Recuperar contexto (RAG) + 2) Llamar LLM + 3) Validar salida
    # Aquí iría tu orquestación; retornamos un ejemplo fijo
    return ChatResponse(text=f"Hola {req.user_id}, recibí: {req.message}")
```

### Streaming (SSE/WebSocket)

- **SSE**: simple, unidireccional; ideal para *token streaming*.
- **WebSocket**: bidireccional; útil para acciones interactivas.

**Ejemplo SSE (FastAPI)**

```python
from fastapi import Request
from fastapi.responses import StreamingResponse
import asyncio

@app.get("/v1/stream")
async def stream(req: Request, q: str):
    async def event_generator():
        for token in ["Hola ", "mundo", "!\n"]:
            yield f"data: {token}\n\n"
            await asyncio.sleep(0.03)
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

### Versionado, CORS y seguridad

- **CORS**: limita orígenes; usa `allow_origins` específicos.
- **Auth**: API keys/JWT; revisa *scopes* por ruta.
- **Rate limiting**: por IP/usuario/clave; responde `429` con `Retry-After`.

---

## Optimización de modelos para menor costo/latencia

### Optimización de prompts y respuestas

- Reduce **tokens**: plantillas concisas, contexto mínimo viable (MMR, filtros de metadatos).
- Usa **salida estructurada** (JSON Schema) para evitar *back-and-forth*.
- **Streaming** para percepción de rapidez.

### Caching y batching

- **Cache** de prompts y respuestas (Redis) por `{hash(prompt+contexto)}`.
- **Cache** de embeddings y resultados de RAG.
- **Batching**: agrupa solicitudes para el servidor de inferencia (p. ej., vLLM/TGI); balancea con latencia.

### Servidores de inferencia y cuantización

- **Servidores**: vLLM, TGI, llama.cpp; ofrecen *throughput* alto, paginación de KV-cache y *tensor parallel*.
- **Cuantización**: 8/4 bits (bitsandbytes/GGUF) para reducir memoria a costa de ligera pérdida de calidad.
- **Distillation/LoRA**: modelos pequeños para tareas frecuentes + *fallback* a modelos grandes según confianza.

### RAG eficiente y vector stores

- **Chunking** 200–400 tokens + solapamiento.
- **Híbrida**: BM25 + vectorial; re‑ranking cuando la precisión lo justifique.
- Indexa con **HNSW/IVF** y ajusta `efSearch/nprobe`.
- **TTL** y *freshness* en metadatos; evita inyectar contexto con similitud baja.

---

## Monitoreo y logging

### Logs estructurados y trazas

- Emite **JSON logs** con `request_id`, `user_id`, latencia, tokens, *tool calls* y errores.
- **OpenTelemetry** para trazas de extremo a extremo.

**Ejemplo (structlog + middleware)**

```python
# app/observability.py
import time, structlog
from starlette.middleware.base import BaseHTTPMiddleware

log = structlog.get_logger()

class Timing(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        t0 = time.time()
        resp = await call_next(request)
        log.info("http_request", path=request.url.path, status=resp.status_code,
                 ms=int((time.time()-t0)*1000))
        return resp
```

**Métricas Prometheus**

```python
from prometheus_client import Counter, Histogram

REQS = Counter("agent_requests_total", "Solicitudes por endpoint", ["route"])
LAT = Histogram("agent_latency_ms", "Latencia en ms", buckets=(50,100,200,500,1000,2000))

@app.middleware("http")
async def metrics_mw(request, call_next):
    import time
    t0 = time.time()
    resp = await call_next(request)
    REQS.labels(route=request.url.path).inc()
    LAT.observe((time.time()-t0)*1000)
    return resp
```

### Métricas de negocio y técnicas

- **Negocio**: tasa de resolución, NPS/CSAT, *deflection rate*.
- **Técnicas**: p50/p95/p99, errores 4xx/5xx, tokens, *cache hit*.

### Alertas y SLOs

- Define **SLOs** (disp., latencia p95). Crea alertas por **error budget** y *burn rate*.

---

## Integración con aplicaciones web o móviles

### Frontends web (Next.js/React)

- **SSE** para streaming.
- Reintentos exponenciales y cancelación (AbortController).

**Fetch SSE minimal**

```ts
// client/sse.ts
export async function stream(url: string, onChunk: (t: string)=>void) {
  const resp = await fetch(url, { headers: { Accept: "text/event-stream" } });
  const reader = resp.body!.getReader();
  const dec = new TextDecoder();
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    onChunk(dec.decode(value));
  }
}
```

### Móvil (React Native/Flutter)

- Usa **WebSocket** para bidireccional; en iOS contempla límites de *background*.
- Minimiza tamaño de payload y evita grandes históricos por solicitud.

### Autenticación y rate limiting

- **JWT** corto + *refresh token*; o API keys por app.
- Rate limit por IP/usuario/app; desacopla del plan de facturación.

---

## Despliegue y operación

### Dockerfile y contenedorización

**Multi‑stage** para imágenes pequeñas.

```dockerfile
# Dockerfile
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app

FROM base AS run
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
```

`.dockerignore`

```
__pycache__/
*.pyc
.env
.git
```

### Ejecución y escalado

- **Uvicorn**: ajusta `--workers` y *threads* según CPU.
- **Autoscaling** (HPA/KEDA): escala por CPU, latencia o cola.
- **Blue/Green** o **Canary** para despliegues seguros.

### Entornos, secretos y migraciones

- **.env** para local; **Vault/Secrets Manager** en prod.
- Migraciones de esquemas (DB/vector store) con *scripts* versionados.

---

## Buenas prácticas y anti‑patrones

**Buenas prácticas**

- Diseña primero los **contratos** (OpenAPI) y *tests* de contrato.
- **Validación** estricta de entrada/salida (Pydantic/JSON Schema).
- **Circuit breakers** y *retries* (con *jitter*).
- **Fallbacks**: a modelo más grande o a humano.
- **Data governance**: PII, retención/TTL, derecho al olvido.
- **Chaos testing** en pre‑producción.

**Anti‑patrones**

- Mezclar lógica de orquestación con controladores HTTP.
- Inyectar **todo** el historial en cada llamada (costos).
- Carecer de **routing** por confianza/uso (mismo modelo para todo).
- No monitorear tokens/latencia (sorpresas de factura).

---

## Checklist de producción

-

---

## Librerías recomendadas

- **fastapi** o **flask** → API para el agente.
- **uvicorn** → servidor ASGI de alto rendimiento.
- **docker** → contenedorización.
- **pydantic** → validación de contratos.
- **httpx/requests** → llamadas a APIs externas.
- **structlog/loguru** → logging estructurado.
- **prometheus\_client** y **opentelemetry‑sdk** → métricas y trazas.
- **redis** → cache; **celery/rq** → trabajos asíncronos.

---

## Apéndices de código

### A1. `requirements.txt`

```
fastapi
uvicorn[standard]
pydantic
structlog
prometheus_client
python-dotenv
httpx
```

### A2. `compose.yaml` (opcional)

```yaml
services:
  api:
    build: .
    ports: ["8080:8080"]
    env_file: .env
    depends_on: [redis]
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
```

### A3. *Middleware* de CORS y seguridad mínima

```python
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tu-frontend.com"],
    allow_credentials=True,
    allow_methods=["POST","GET"],
    allow_headers=["Authorization","Content-Type"],
)
```

### A4. Salud y *readiness*

```python
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/readyz")
def readyz():
    # Comprobar conectividad a vector store, cache, LLM, etc.
    return {"ready": True}
```


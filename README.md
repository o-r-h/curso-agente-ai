# curso-agente-ai
---

## 📚 **Plan de Estudios – Creación de Agentes de IA en Python**

---

### **Nivel 1 – Fundamentos de IA conversacional**

**Objetivo:** Comprender qué es un agente, cómo se diferencia de un simple chatbot y cómo interpretar entradas del usuario.
**Temas:**

* Qué es un agente de IA y tipos (reactivo, proactivo, autónomo).
* Procesamiento de lenguaje natural (**NLP**) básico.
* Limpieza y tokenización de texto.
* Concepto de *intención* y *entidades*.
* Introducción a modelos pre-entrenados.

**Librerías recomendadas:**

* `nltk` → tokenización, stemming, stopwords.
* `spaCy` → procesamiento más avanzado y detección de entidades.
* `scikit-learn` → clasificación básica de intenciones.

---

### **Nivel 2 – Detección de intenciones y clasificación**

**Objetivo:** Crear un modelo que pueda identificar qué quiere el usuario.
**Temas:**

* Preparación de datasets de intenciones (JSON, CSV).
* Representación del texto (*Bag of Words*, TF-IDF, embeddings).
* Modelos clásicos de clasificación: Naive Bayes, SVM, Logistic Regression.
* Uso de embeddings semánticos.

**Librerías recomendadas:**

* `scikit-learn` → modelos clásicos.
* `sentence-transformers` → embeddings como *all-MiniLM*.
* `pandas` → manejo de datasets.

---

### **Nivel 3 – Memoria en agentes**

**Objetivo:** Que el agente recuerde el contexto y lo use en la conversación.
**Temas:**

* Tipos de memoria: de corto plazo (contexto inmediato) y de largo plazo (histórico).
* Representación del contexto en vectores.
* Bases de datos vectoriales (*Vector Stores*).
* Recuperación de contexto relevante (*Retrieval*).

**Librerías recomendadas:**

* `langchain` → manejo de memoria conversacional.
* `faiss` o `chromadb` → almacenamiento y búsqueda vectorial.
* `sqlite3` (opcional) → persistencia simple.

---

### **Nivel 4 – Integración con modelos de lenguaje**

**Objetivo:** Usar LLMs para responder y tomar decisiones.
**Temas:**

* Diferencia entre modelos locales y APIs (OpenAI, HuggingFace).
* Uso de prompts y *prompt engineering*.
* Control de salida y reducción de alucinaciones.
* Sistemas RAG (*Retrieval-Augmented Generation*).

**Librerías recomendadas:**

* `openai` o `transformers` → para conectarte a LLMs.
* `langchain` → orquestación y conexión con memoria.
* `transformers` → para modelos locales (ej. LLaMA, Mistral).

---

### **Nivel 5 – Herramientas y acciones del agente**

**Objetivo:** Que el agente pueda hacer cosas más allá de responder texto.
**Temas:**

* Herramientas (*tools*) y *function calling*.
* Integración con APIs externas.
* Ejecución de tareas (buscar en Google, consultar una base de datos).
* Manejo de múltiples herramientas y decisiones de uso.

**Librerías recomendadas:**

* `langchain` → definición y ejecución de herramientas.
* `requests` → llamadas a APIs externas.
* `python-dotenv` → gestión de credenciales.

---

### **Nivel 6 – Agentes autónomos**

**Objetivo:** Crear agentes que planifican y actúan de forma autónoma.
**Temas:**

* Arquitecturas de agentes (ReAct, AutoGPT, BabyAGI).
* Planificación paso a paso (*planning*).
* Control de bucles y límites.
* Evaluación y depuración de agentes.

**Librerías recomendadas:**

* `langgraph` o `crewai` → agentes multi-herramienta y multi-paso.
* `langchain-experimental` → prototipado rápido de agentes.

---

### **Nivel 7 – Escalabilidad y despliegue**

**Objetivo:** Llevar tu agente a producción.
**Temas:**

* APIs REST para exponer el agente.
* Optimización de modelos para menor costo/latencia.
* Monitoreo y logging.
* Integración con aplicaciones web o móviles.

**Librerías recomendadas:**

* `fastapi` o `flask` → API para el agente.
* `uvicorn` → servidor ASGI.
* `docker` → contenedorización.

---

💡 **Sugerencia de orden de aprendizaje:**

1. Fundamentos y NLP básico.
2. Clasificación de intenciones.
3. Memoria y contexto.
4. Integración con LLMs.
5. Herramientas.
6. Agentes autónomos.
7. Despliegue.
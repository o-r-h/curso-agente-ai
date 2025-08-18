# Nivel 1 – Fundamentos de IA conversacional

> **Objetivo**: Comprender qué es un agente conversacional, en qué se diferencia de un simple chatbot, y cómo interpretar las entradas del usuario mediante conceptos básicos de NLP (intenciones y entidades) y modelos pre‑entrenados.

---

## Tabla de contenidos
1. [¿Qué es un agente conversacional?](#qué-es-un-agente-conversacional)
2. [Tipos de agentes: reactivo, proactivo, autónomo](#tipos-de-agentes-reactivo-proactivo-autónomo)
3. [Arquitectura mínima de un agente](#arquitectura-mínima-de-un-agente)
4. [NLP básico para conversaciones](#nlp-básico-para-conversaciones)
   - [Limpieza y normalización](#limpieza-y-normalización)
   - [Tokenización](#tokenización)
   - [Stemming vs lematización](#stemming-vs-lematización)
   - [Stopwords y puntuación](#stopwords-y-puntuación)
5. [Intenciones y entidades](#intenciones-y-entidades)
   - [Diseño de intenciones](#diseño-de-intenciones)
   - [Diseño de entidades (slots)](#diseño-de-entidades-slots)
   - [Desambiguación y contexto](#desambiguación-y-contexto)
6. [Introducción a modelos pre‑entrenados](#introducción-a-modelos-pre-entrenados)
   - [Embeddings clásicos vs transformadores](#embeddings-clásicos-vs-transformadores)
   - [Zero‑shot, few‑shot y fine‑tuning](#zero-shot-few-shot-y-fine-tuning)
7. [Clasificación básica de intenciones con scikit‑learn](#clasificación-básica-de-intenciones-con-scikit-learn)
8. [Extracción de entidades con spaCy](#extracción-de-entidades-con-spacy)
9. [Evaluación](#evaluación)
10. [Buenas prácticas y anti‑patrones](#buenas-prácticas-y-anti-patrones)
11. [Checklists](#checklists)
12. [Glosario](#glosario)
13. [Referencias y librerías recomendadas](#referencias-y-librerías-recomendadas)

---

## ¿Qué es un agente conversacional?
Un **agente** es un sistema que **percibe**, **razona** y **actúa** para alcanzar objetivos. En el contexto conversacional, el agente:
- *Percibe*: recibe texto/voz del usuario (entrada) y el estado del entorno (contexto, historial, herramientas disponibles).
- *Razona*: interpreta la intención, extrae entidades, consulta conocimiento o herramientas (APIs, bases de datos) y decide un plan.
- *Actúa*: genera una respuesta y ejecuta acciones (p. ej., reservar, consultar inventario, iniciar un flujo).

**Diferencia con un “simple chatbot”**:
- Un chatbot tradicional suele ser **scripted** (reglas/árboles de decisión). Un agente moderno combina NLU (comprensión), razonamiento y **capacidad de actuar** sobre herramientas externas.
- El agente mantiene **estado**, usa **políticas** para decidir (p. ej., priorizar seguridad) y puede **aprender**/adaptarse.

## Tipos de agentes: reactivo, proactivo, autónomo
- **Reactivo**: responde a eventos/consultas; no inicia interacción por sí mismo. Adecuado para FAQ, soporte nivel 1.
- **Proactivo**: toma la iniciativa cuando detecta oportunidades o riesgos (recordatorios, alertas, sugerencias contextuales).
- **Autónomo**: persigue objetivos con planificación multi‑paso, invoca herramientas y se autoevalúa. Requiere límites claros (scope, costos, seguridad).

> **Ejemplo rápido**: Un agente de viajes
> - *Reactivo*: responde precios cuando se le pregunta.
> - *Proactivo*: avisa que el precio bajó.
> - *Autónomo*: busca vuelos, compara, reserva según preferencias y presupuesto, verificando restricciones.

## Arquitectura mínima de un agente
```
Usuario → Preprocesamiento → NLU (intención/entidades) → Orquestación/Política → Acciones/Herramientas → NLG/Respuesta → Usuario
                              ↑——————————— Memoria / Estado / Contexto ————————————↑
```
**Componentes clave**:
- **Preprocesamiento**: normaliza texto (tildes, mayúsculas, emojis, URLs, ruido).
- **NLU**: clasificación de *intenciones* y extracción de *entidades*.
- **Orquestación**: decide el próximo paso (p. ej., pedir datos faltantes, llamar API, transferir a humano).
- **Memoria/Estado**: guarda slots, historial conversacional y preferencias.
- **NLG**: transforma la decisión en lenguaje natural (plantillas o modelos generativos).

## NLP básico para conversaciones
### Limpieza y normalización
- Lowercasing (cuando no importe el caso).
- Quitar/normalizar tildes, emojis, URLs, @menciones, #hashtags.
- Corrección ortográfica ligera o normalización de argot si afecta la intención.
- Manejo de idiomas mixtos (code‑switching) y variantes regionales.

### Tokenización
- Separar texto en **tokens** (palabras, subpalabras). En español, cuida contracciones y nombres compuestos.

### Stemming vs lematización
- **Stemming**: recorta sufijos (rápido pero tosco).  
- **Lematización**: reduce a la forma canónica (mejor semántica; soportado por spaCy).

### Stopwords y puntuación
- Quitar palabras muy comunes (*de, la, y*), si mejoran la señal.  
- Mantener puntuación a veces ayuda para **intención** (¡ vs ?), pero normalmente se elimina para **bolsas de palabras**.

## Intenciones y entidades
**Intención** = objetivo del usuario (p. ej., *comprar_boletos*).  
**Entidades** = piezas clave (p. ej., *ciudad_origen=CDMX*, *fecha=2025‑08‑21*).

### Diseño de intenciones
- Mantén **mutuamente exclusivas** y **colectivamente exhaustivas** dentro del dominio.
- Evita intenciones demasiado granulares; usa **slots** para variaciones.
- Recoge ejemplos reales (frases de usuarios) y **paráfrasis**.

### Diseño de entidades (slots)
- Tipos: **fechas**, **números**, **moneda**, **ubicaciones**, **personas**, **productos**, **categorías**.
- Fuentes: reglas/regex, diccionarios, NER con modelos (spaCy), o híbridos.

### Desambiguación y contexto
- Si falta un slot obligatorio, **repregunta** de forma específica.  
- Si hay ambigüedad (dos posibles ciudades), **confirma** antes de actuar.  
- Usa el **historial** (turnos previos) para completar slots implícitos.

## Introducción a modelos pre‑entrenados
### Embeddings clásicos vs transformadores
- **Clásicos**: Bag‑of‑Words, TF‑IDF, *n‑gramas* (simples, eficientes, base para clasificadores lineales).
- **Neurales**: Word2Vec, GloVe, FastText (capturan similitud, pero son estáticos).
- **Transformadores** (BERT, RoBERTa, DistilBERT, sentence‑BERT): contextuales, excelente desempeño en NLU.

### Zero‑shot, few‑shot y fine‑tuning
- **Zero‑shot**: usar un modelo general para clasificar sin re‑entrenar (útil para bootstrap).  
- **Few‑shot**: proporcionar pocos ejemplos de cada intención.  
- **Fine‑tuning**: ajustar el modelo con datos del dominio (mejor performance, más costo).

## Clasificación básica de intenciones con scikit‑learn
> **Objetivo**: construir un clasificador sencillo (TF‑IDF + Regresión Logística) como baseline.

### Instalación
```bash
pip install nltk spacy scikit-learn
python -m spacy download es_core_news_sm
```

### Datos de ejemplo
```python
corpus = [
    ("quiero reservar un vuelo a madrid", "reservar_vuelo"),
    ("busca hoteles en barcelona", "buscar_hotel"),
    ("necesito cambiar mi boleto", "cambiar_boleto"),
    ("hay vuelos baratos a lima?", "buscar_vuelo"),
    ("cancela mi reservación de hotel", "cancelar_reserva"),
    ("me ayudas a comprar un vuelo", "reservar_vuelo"),
    ("encuentra un hotel 5 estrellas", "buscar_hotel"),
]
```

### Pipeline clásico (TF‑IDF → clasificador)
```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

X, y = zip(*corpus)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
    ("clf", LogisticRegression(max_iter=1000))
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred))

# Inferencia
nueva_frase = "puedes reservarme un vuelo a querétaro?"
print(pipe.predict([nueva_frase])[0])
```

> **Notas**: 
> - Con pocos datos, estratifica y usa *k‑fold*.  
> - Prueba SVM lineal o Naive Bayes como alternativas rápidas.

## Extracción de entidades con spaCy
```python
import spacy
nlp = spacy.load("es_core_news_sm")
texto = "Reserva un hotel en Ciudad de México del 20 al 22 de septiembre por 2000 MXN"
doc = nlp(texto)
for ent in doc.ents:
    print(ent.text, ent.label_)
```

**Patrones personalizados** con `Matcher`/`EntityRuler` para dominios específicos (p. ej., códigos de aeropuerto):
```python
from spacy.pipeline import EntityRuler
ruler = nlp.add_pipe("entity_ruler", before="ner")
patterns = [
    {"label": "AEROPUERTO", "pattern": "CDMX"},
    {"label": "MONEDA", "pattern": [{"TEXT": {"REGEX": "^(MXN|USD|EUR)$"}}]},
]
ruler.add_patterns(patterns)
```

## Evaluación
- **Clasificación de intenciones**: precisión, recall, F1 por clase, *macro F1* y matriz de confusión.
- **Entidades**: *precision*, *recall*, *F1* a nivel entidad (exact match), más análisis de errores por tipo.
- **Conversación**: tasa de resolución, handoffs a humano, NPS/CSAT, latencia, tasa de caídas.

## Buenas prácticas y anti‑patrones
**Buenas prácticas**
- Empieza con un **baseline** interpretable (TF‑IDF + lineal). Escala a transformadores cuando el caso lo pida.
- Diseña intenciones con **datos reales** y revisiones periódicas; agrega un fallback (p. ej., `no_entendida`).
- Usa **validación cruzada** y *hold‑out* por usuario para evitar fuga de información.
- Para entidades, combina **NER + reglas**; versiona diccionarios y pruebas.
- Incluye **seguridad y privacidad**: PII, mascarado/logging mínimo necesario, controles de abuso.
- Añade **telemetría** accionable y bucle de mejora continua.

**Anti‑patrones**
- Demasiadas intenciones casi idénticas → confusiones y datos escasos.
- Confiar solo en NER genérico para dominios especializados → patrones fallan.
- Ignorar el **contexto** (historial) o los **slots obligatorios**.
- No etiquetar correctamente el conjunto de prueba (gold) → métricas engañosas.

## Checklists
**Antes de entrenar**
- [ ] Intenciones definidas y no solapadas.
- [ ] Slots obligatorios y opcionales identificados.
- [ ] Conjunto de ejemplos por intención ≥ 20–50 (idealmente).
- [ ] Política de fallback y handoff a humano.

**Durante el desarrollo**
- [ ] Pipeline reproducible (semillas, versiones, artefactos).
- [ ] Validación cruzada y *error analysis*.
- [ ] Pruebas unitarias para extractores de entidades.

**Antes de producción**
- [ ] Telemetría: métricas de NLU y negocio.
- [ ] Límites de seguridad (PII, contenido tóxico, *rate limits*).
- [ ] Playbooks de escalamiento a humano.

## Glosario
- **Agente**: sistema que percibe, razona y actúa para lograr objetivos.
- **Chatbot**: interfaz conversacional, tradicionalmente basada en reglas.
- **NLU**: comprensión del lenguaje natural (intenciones, entidades).
- **NLG**: generación de lenguaje natural (respuestas).
- **Slot**: campo/atributo requerido para completar una tarea (p. ej., fecha).
- **Fallback**: manejo de consultas no entendidas o fuera de alcance.
- **Embedding**: representación numérica de texto.

## Referencias y librerías recomendadas
- **NLTK**: tokenización, stemming, stopwords.  
- **spaCy**: procesamiento avanzado, NER, *EntityRuler*.  
- **scikit‑learn**: clasificación (SVM, Regresión Logística, Naive Bayes) y pipelines.  
- **Otros útiles**: regex para reglas, `dateparser` para fechas, `pydantic` para validar slots.

---

### Apéndice: ejemplo completo (mini‑proyecto)
```python
# 1) Datos simples
datos = [
  ("reserva vuelo a madrid mañana", "reservar_vuelo"),
  ("busca hotel en barcelona", "buscar_hotel"),
  ("cambia mi boleto a la tarde", "cambiar_boleto"),
  ("vuelos baratos a lima", "buscar_vuelo"),
  ("cancela mi reserva de hotel", "cancelar_reserva"),
]

# 2) Entrenamiento baseline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

X, y = zip(*datos)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
modelo = Pipeline([
  ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
  ("clf", LogisticRegression(max_iter=1000))
])
modelo.fit(Xtr, ytr)

# 3) Extracción de entidades con spaCy
import spacy
nlp = spacy.load("es_core_news_sm")
texto = "reserva un hotel en Ciudad de México del 20 al 22 de septiembre"
doc = nlp(texto)
slots = {ent.label_: ent.text for ent in doc.ents}

# 4) Política simple
intencion = modelo.predict([texto])[0]
requeridos = {
  "reservar_vuelo": ["LOC", "DATE"],
  "buscar_hotel": ["LOC", "DATE"],
}
pendientes = [s for s in requeridos.get(intencion, []) if s not in slots]
if pendientes:
  print(f"Me falta: {pendientes}. ¿Podrías compartir esos datos?")
else:
  print(f"OK, procedo con {intencion} usando {slots}")
```

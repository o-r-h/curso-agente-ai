# Nivel 2 – Detección de intenciones y clasificación

> **Objetivo**: Crear un modelo que pueda identificar qué quiere el usuario (clasificación de intenciones), usando técnicas clásicas de ML y representaciones semánticas.

---

## Tabla de contenidos
1. [Preparación de datasets de intenciones](#preparación-de-datasets-de-intenciones)
   - [Formato JSON](#formato-json)
   - [Formato CSV](#formato-csv)
   - [Ejemplo de dataset](#ejemplo-de-dataset)
2. [Representación del texto](#representación-del-texto)
   - [Bag of Words (BoW)](#bag-of-words-bow)
   - [TF‑IDF](#tf-idf)
   - [Embeddings semánticos](#embeddings-semánticos)
3. [Modelos clásicos de clasificación](#modelos-clásicos-de-clasificación)
   - [Naive Bayes](#naive-bayes)
   - [SVM](#svm)
   - [Regresión Logística](#regresión-logística)
4. [Uso de embeddings semánticos](#uso-de-embeddings-semánticos)
5. [Evaluación y métricas](#evaluación-y-métricas)
6. [Buenas prácticas](#buenas-prácticas)
7. [Checklist de implementación](#checklist-de-implementación)
8. [Glosario](#glosario)
9. [Referencias y librerías recomendadas](#referencias-y-librerías-recomendadas)

---

## Preparación de datasets de intenciones
Un **dataset de intenciones** contiene ejemplos de frases de usuarios y la etiqueta correspondiente (intención). Es la base para entrenar modelos de clasificación.

### Formato JSON
```json
{
  "intents": [
    {
      "tag": "saludo",
      "patterns": ["hola", "buenos días", "qué tal"]
    },
    {
      "tag": "despedida",
      "patterns": ["adiós", "nos vemos", "hasta luego"]
    },
    {
      "tag": "reservar_vuelo",
      "patterns": ["quiero reservar un vuelo", "necesito un boleto a madrid"]
    }
  ]
}
```

### Formato CSV
```csv
texto,intencion
"hola",saludo
"adiós",despedida
"quiero reservar un vuelo a madrid",reservar_vuelo
"me ayudas a encontrar un hotel",buscar_hotel
```

### Ejemplo de dataset
- Al menos **20‑50 ejemplos por intención**.
- Variar con sinónimos, diferentes estructuras, errores comunes de escritura.
- Dividir en *train/test* de manera estratificada.

## Representación del texto
### Bag of Words (BoW)
- Representa cada frase como un vector binario/contado de palabras.
- Simple pero ignora orden y semántica.

### TF‑IDF
- Extiende BoW ponderando palabras frecuentes en un documento pero raras en el corpus.
- Mejora la discriminación de términos relevantes.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = ["quiero reservar un vuelo", "hola", "busca hoteles"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out())
print(X.toarray())
```

### Embeddings semánticos
- Representaciones densas y continuas que capturan significado y contexto.
- Modelos como **all‑MiniLM‑L6‑v2** (Sentence‑Transformers) generan vectores de 384 dimensiones.

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = ["quiero reservar un vuelo", "hola"]
embeddings = model.encode(sentences)
print(embeddings.shape)  # (2, 384)
```

## Modelos clásicos de clasificación
### Naive Bayes
- Asume independencia condicional entre palabras.
- Eficiente y funciona bien en textos cortos con BoW/TF‑IDF.

```python
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train, y_train)
```

### SVM
- Encuentra un hiperplano que separa clases con máximo margen.
- Buen rendimiento en alta dimensionalidad (TF‑IDF).

```python
from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(X_train, y_train)
```

### Regresión Logística
- Modelo lineal probabilístico, interpretable.
- Usado ampliamente como baseline.

```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
```

## Uso de embeddings semánticos
- Generar embeddings con **Sentence‑Transformers**.
- Usar clasificadores clásicos o capas densas simples (MLP) sobre embeddings.

```python
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

# Embeddings
dataset = ["hola", "adiós", "quiero reservar un vuelo"]
labels = ["saludo", "despedida", "reservar_vuelo"]
model = SentenceTransformer('all-MiniLM-L6-v2')
X = model.encode(dataset)

# Clasificación
clf = LogisticRegression(max_iter=1000)
clf.fit(X, labels)
print(clf.predict(model.encode(["me ayudas a comprar un boleto a lima"])))
```

## Evaluación y métricas
- **Accuracy**: % de predicciones correctas.
- **Precision, Recall, F1**: más útiles cuando las clases están desbalanceadas.
- **Matriz de confusión**: muestra errores comunes.
- **Cross‑validation**: reduce riesgo de sobreajuste.

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

## Buenas prácticas
- Empieza con TF‑IDF + Regresión Logística como baseline.
- Usar embeddings semánticos si hay frases más complejas o sinónimos.
- Ampliar dataset con **paráfrasis** y errores de usuario.
- Normalizar/limpiar texto (minúsculas, tildes, caracteres especiales).
- Dividir datos en **train/dev/test**.
- Mantener un `fallback` para intenciones desconocidas.

## Checklist de implementación
- [ ] Dataset con ≥ 20 ejemplos por intención.
- [ ] Preprocesamiento de texto documentado.
- [ ] Vectorización (BoW, TF‑IDF, embeddings) probada.
- [ ] Baseline entrenado (Logistic Regression o SVM).
- [ ] Embeddings semánticos implementados (opcional).
- [ ] Evaluación con métricas claras.
- [ ] Código reproducible con semillas y versiones.

## Glosario
- **Intención**: lo que el usuario quiere lograr (ej. `reservar_vuelo`).
- **BoW**: representación de texto como bolsa de palabras.
- **TF‑IDF**: ponderación de términos para resaltar los más relevantes.
- **Embedding**: vector denso que captura semántica.
- **SVM**: clasificador de margen máximo.
- **Naive Bayes**: clasificador probabilístico simple.

## Referencias y librerías recomendadas
- **scikit‑learn**: clasificación (Naive Bayes, SVM, Logistic Regression).
- **sentence‑transformers**: embeddings pre‑entrenados (ej. `all‑MiniLM‑L6‑v2`).
- **pandas**: manejo de datasets en CSV/JSON.
- **numpy**: manipulación de vectores.

---

### Apéndice: pipeline completo con TF‑IDF + Logistic Regression
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Dataset
corpus = [
  ("hola", "saludo"),
  ("buenos días", "saludo"),
  ("adiós", "despedida"),
  ("quiero reservar un vuelo a madrid", "reservar_vuelo"),
  ("busca hoteles en barcelona", "buscar_hotel"),
]
X, y = zip(*corpus)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Pipeline
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=1000))
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred))
```

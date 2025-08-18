# Nivel 5 – Herramientas y acciones del agente

> **Objetivo**: Que el agente pueda realizar **acciones más allá de responder texto**, integrándose con APIs, ejecutando consultas externas y decidiendo qué herramienta usar en cada caso.

---

## Tabla de contenidos

1. [Introducción a herramientas en agentes](#introducción-a-herramientas-en-agentes)
2. [Concepto de ](#concepto-de-function-calling)[*function calling*](#concepto-de-function-calling)
3. [Integración con APIs externas](#integración-con-apis-externas)
4. [Ejecución de tareas prácticas](#ejecución-de-tareas-prácticas)
   - [Búsqueda en la web](#búsqueda-en-la-web)
   - [Consulta a base de datos](#consulta-a-base-de-datos)
   - [Servicios de terceros (ej. clima, finanzas, soporte)](#servicios-de-terceros-ej-clima-finanzas-soporte)
5. [Manejo de múltiples herramientas](#manejo-de-múltiples-herramientas)
6. [Implementación práctica](#implementación-práctica)
   - [Definición de herramientas con LangChain](#definición-de-herramientas-con-langchain)
   - [Uso de ](#uso-de-requests-para-apis)[`requests`](#uso-de-requests-para-apis)[ para APIs](#uso-de-requests-para-apis)
   - [Gestión de credenciales con dotenv](#gestión-de-credenciales-con-dotenv)
7. [Buenas prácticas](#buenas-prácticas)
8. [Checklist de implementación](#checklist-de-implementación)
9. [Glosario](#glosario)
10. [Referencias y librerías recomendadas](#referencias-y-librerías-recomendadas)

---

## Introducción a herramientas en agentes

Un **agente inteligente** debe ser capaz de **actuar en el mundo externo**, no solo responder texto. Para ello se definen *herramientas* (tools): funciones que el modelo puede invocar de manera controlada.

Ejemplo: el agente puede llamar a una API de clima para responder *“¿Necesito paraguas mañana en Monterrey?”* en lugar de inventar la respuesta.

## Concepto de *function calling*

- **Definición**: mecanismo mediante el cual el LLM **decide invocar** una función externa con ciertos parámetros.
- **Flujo típico**:
  1. Usuario hace una consulta.
  2. El LLM detecta que debe usar una herramienta.
  3. El agente ejecuta la función con parámetros generados.
  4. El resultado se inyecta de nuevo en la conversación.

Ejemplo simplificado:

```json
{
  "name": "get_weather",
  "arguments": {"city": "Monterrey"}
}
```

## Integración con APIs externas

- APIs REST/GraphQL permiten extender las capacidades del agente.
- Librerías como `requests` facilitan las llamadas HTTP.
- Consideraciones:
  - **Autenticación**: tokens/keys seguros.
  - **Tiempos de espera** y *retries*.
  - **Parsing** de respuestas (JSON, XML).
  - **Validación** antes de mostrar al usuario.

## Ejecución de tareas prácticas

### Búsqueda en la web

- Conexión a motores de búsqueda o APIs de terceros (ej. Google Custom Search, Bing API).

### Consulta a base de datos

- Acceso directo a SQL o mediante servicios intermedios.
- Ejemplo: `sqlite3` para prototipos.

### Servicios de terceros (ej. clima, finanzas, soporte)

- Conectar con APIs públicas o privadas.
- Ejemplo: OpenWeather, Stripe, Zendesk.

## Manejo de múltiples herramientas

- El agente puede tener un **catálogo de herramientas**.
- Estrategias de decisión:
  - **Model-driven**: el LLM elige directamente la herramienta.
  - **Rule-driven**: reglas o *routers* definen qué herramienta usar.
  - **Híbrido**: mezcla de heurísticas + decisión del modelo.

**Desafíos**:

- Resolución de conflictos (dos herramientas aplicables).
- Gestión de errores de ejecución.
- Priorización de latencia y costos.

## Implementación práctica

### Definición de herramientas con LangChain

```python
from langchain.agents import Tool

# Definir una herramienta de búsqueda simulada
def search_web(query: str) -> str:
    return f"Resultados de búsqueda para: {query}"

herramienta = Tool(
    name="busqueda_web",
    func=search_web,
    description="Usa esta herramienta para buscar información en la web"
)
```

### Uso de `requests` para APIs

```python
import requests

API_KEY = "TU_API_KEY"
CITY = "Monterrey"

url = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
resp = requests.get(url, timeout=10)
data = resp.json()

print(f"Temperatura en {CITY}: {data['main']['temp']}°C")
```

### Gestión de credenciales con dotenv

```python
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")
```

## Buenas prácticas

- Definir **interfaces claras** para herramientas.
- Manejar errores y caídas de servicios.
- Limitar el número de herramientas activas en producción.
- Monitorear uso de APIs externas (latencia, costos, límites).
- Usar logs para depurar interacciones del agente.

## Checklist de implementación

-

## Glosario

- **Tool (herramienta)**: función externa que un agente puede invocar.
- **Function calling**: mecanismo que permite al LLM decidir ejecutar funciones.
- **API**: interfaz para interactuar con servicios externos.
- **Router**: componente que decide qué herramienta se debe usar.

## Referencias y librerías recomendadas

- **langchain**: orquestación y definición de herramientas.
- **requests**: llamadas HTTP simples y eficientes.
- **python-dotenv**: gestión de variables de entorno.
- **sqlite3**: acceso ligero a bases de datos.


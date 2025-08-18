#!/usr/bin/env bash
set -euo pipefail

# === Setup de entorno para el Notebook de Memoria en Agentes ===
# Uso:
#   chmod +x run.sh
#   ./run.sh
#
# Variables opcionales:
#   OPENAI_API_KEY=sk-... ./run.sh

# 1) Crear venv
python3 -m venv .venv
source .venv/bin/activate

# 2) Instalar dependencias
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) Mostrar si hay API key configurada
python - <<'PY'
import os
print("OPENAI_READY:", bool(os.environ.get("OPENAI_API_KEY")))
PY

# 4) Lanzar Jupyter Lab (sin token, solo para entorno local)
jupyter lab --NotebookApp.token='' --NotebookApp.password='' --no-browser

#!/bin/zsh
# Doble clic para abrir SAM3 Studio en el navegador
cd "$(dirname "$0")"
exec .venv/bin/python app.py "$@" 2> >(grep --line-buffered -v "W:onnxruntime" >&2)

# SAM3 Studio

Interfaz web local para segmentar y seguir objetos en imágenes y vídeos con
**SAM 3** en un Mac con Apple Silicon (CoreML), usando
[osam](https://github.com/wkentaro/osam). Todo se ejecuta en tu ordenador.

*English below.*

## Qué hace

- **Texto:** escribe `person`, `car`, `horse`… y segmenta todos los que haya.
  En imágenes admite varias clases separadas por comas.
- **Clic:** clic sobre algo sin máscara lo añade; clic sobre una máscara la quita.
- **Caja:** dibuja un rectángulo alrededor de un objeto; en vídeo lo sigue en
  todos los fotogramas.
- **Recuento** de objetos por fotograma y por clase.
- **Exportar** a `videos exportados/`:
  - vídeo con las máscaras pintadas (`.mp4`)
  - vídeo original con los mismos fotogramas (`.mp4`)
  - máscara en blanco y negro (ProRes `.mov`)
  - máscaras en color con transparencia (ProRes 4444 `.mov`)
  - una máscara por objeto (ProRes `.mov`), lista para Final Cut Pro, Premiere o DaVinci
- Interfaz en español e inglés.

## Requisitos

- Mac con Apple Silicon y macOS 14 o posterior
- [uv](https://docs.astral.sh/uv/) (o Python 3.11+ con `pip`)
- `ffmpeg` (`brew install ffmpeg`)

## Instalación

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

La primera vez que se abre, osam descarga SAM 3 (unos 1,7 GB) en
`~/.cache/osam`, y CoreML crea una caché de 1,8 GB.

## Uso

Doble clic en `SAM3 Studio.command`, o:

```bash
.venv/bin/python app.py              # abre http://127.0.0.1:8765
.venv/bin/python app.py --device webgpu --port 9000
```

## Límites

- En vídeo, osam sigue **una sola clase de texto** cada vez.
- Velocidad en un M4: unos 4 s por fotograma al seguir objetos a 960×540.
  Usa menos FPS en "Opciones de vídeo" para vídeos largos. Quitar un objeto es
  instantáneo; añadir uno vuelve a recorrer el vídeo.

---

## English

Local web UI to segment and track objects in images and videos with **SAM 3**
on Apple Silicon (CoreML) via [osam](https://github.com/wkentaro/osam).
Find objects by text, click to add or remove masks, draw boxes to track,
count objects per frame, and export the painted video, B/W mattes, alpha
masks and one matte per object (ProRes) for video editors.

Install with the commands above, then run `.venv/bin/python app.py`.
Video tracking takes one text class at a time.

## Licencias / Licenses

El código de este repositorio es MIT. osam es MIT. Los pesos de SAM 3 no se
incluyen: osam los descarga y están bajo la
[SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE) de Meta.

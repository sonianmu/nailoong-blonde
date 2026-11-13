#!/bin/sh
# quick fit loop: full preview (reference/_preview.png) + enlarged face crop (reference/_pv_face.png)
cd "$(dirname "$0")/../.."
.venv/bin/python tools/model/preview.py reference/_preview.png 440

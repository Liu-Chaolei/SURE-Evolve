#!/usr/bin/env bash
echo "=== pip list ==="
/opt/conda/bin/pip list 2>&1 | head -40
echo "=== conda list ==="
/opt/conda/bin/conda list 2>&1 | head -40
echo "=== check pydantic ==="
/opt/conda/bin/python -c "import pydantic; print(pydantic.__version__)" 2>&1
echo "=== pip check index ==="
/opt/conda/bin/pip config list 2>&1
echo "=== DONE ==="

#!/usr/bin/env bash
/opt/conda/bin/python -c "import pynini; print('pynini:', pynini.__version__)" 2>&1
/opt/conda/bin/pip show pynini 2>&1 | head -5
/opt/conda/bin/pip show WeTextProcessing 2>&1 | head -5

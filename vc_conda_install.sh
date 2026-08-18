#!/usr/bin/env bash
export PATH=/opt/conda/bin:$PATH
# Check conda connectivity and install wetext processing
conda install -y -c conda-forge pynini 2>&1 | tail -20
pip install WeTextProcessing 2>&1 | tail -10
python -c "from tn.chinese.normalizer import Normalizer; print('OK')" 2>&1

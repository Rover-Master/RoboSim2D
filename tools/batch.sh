#!/bin/bash
# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
cat manifest.yaml | python3 tools/batch.py data/world $@ 2> batch.log

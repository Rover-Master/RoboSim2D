#!/bin/bash
# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
cat batch.yaml | python3 batch.py data/world 2> batch.log

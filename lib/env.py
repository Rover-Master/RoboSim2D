# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from os import environ

if "SEED" in environ:
    from random import seed

    s = int(environ["SEED"])
    seed(s)

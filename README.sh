#!/bin/bash
# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================

mkdir -p tmp

run() {
python3 -m $1.$2 data/world --prefix=tmp/ \
    --scale=2 --dpi-scale=2 \
    --src=0,-6 \
    --dst=-7.5,2 \
    > /dev/null
mv tmp/$2.png doc/$2.png || echo "Failed to run $1.$2"
}

run simulation RandomWalk; mv doc/RandomWalk.png doc/RandomWalk-01.png
run simulation RandomWalk; mv doc/RandomWalk.png doc/RandomWalk-02.png
run simulation RandomWalk; mv doc/RandomWalk.png doc/RandomWalk-03.png
run simulation RandomWalk; mv doc/RandomWalk.png doc/RandomWalk-04.png
run simulation Bug0L
run simulation Bug0R
run simulation Bug1L
run simulation Bug1R
run simulation Bug2L
run simulation Bug2R

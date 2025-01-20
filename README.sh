#!/bin/bash
mkdir -p tmp

run() {
python3 -m $1.$2 data/world --prefix=tmp/ \
    --scale=2 --dpi-scale=2 \
    --src=-3.60,-2.10 \
    --dst=0.80,-6.57 \
    > /dev/null
mv tmp/$2.png doc/$2.png || echo "Failed to run $1.$2"
}

run simulation RandomWalk
run simulation Bug0L
run simulation Bug0R
run simulation Bug1L
run simulation Bug1R
run simulation Bug2L
run simulation Bug2R

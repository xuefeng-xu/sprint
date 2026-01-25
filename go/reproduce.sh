#!/bin/bash

for i in {1..3}
do
    for beamw in {2..30..2}
    do
        go run search.go -dataset $1 -beamw $beamw
    done
done

python plot.py --dataset $1

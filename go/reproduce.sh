#!/bin/bash

for i in {1..3}
do
    for beamw in {2..30..2}
    do
        go run search.go -dataset $1 -beamw $beamw
    done
done

for i in {1..3}
do
    for beamw in {2..30..2}
    do
        go run batch/main.go -dataset $1 -batchsize $2 -beamw $beamw
    done
done

python plot.py --dataset $1 --batchsize $2

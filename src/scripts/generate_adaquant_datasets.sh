#!/bin/bash

# python src/scripts/make_synthetic_data.py  1 1 10 20000 100 100

# python src/scripts/make_synthetic_data.py  1 1 10 20000 100 100

for NCLIENTS in 10 100 200 400
do
    echo $NCLIENTS
    python src/scripts/make_synthetic_data.py  1 1 10 $(($NCLIENTS * 200)) $NCLIENTS $NCLIENTS
done
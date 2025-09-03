#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

docker volume create stoicalgorithm-output

# Run the algorithm
MEMORY="16g"

docker run --rm --gpus all \
        --memory=$MEMORY --memory-swap=$MEMORY \
        --cap-drop=ALL --security-opt="no-new-privileges" \
        --network none --shm-size=128m --pids-limit 256 \
        -v $SCRIPTPATH/test/:/input/ \
        -v stoicalgorithm-output:/output/ \
        stoicalgorithm

docker run --rm \
        -v stoicalgorithm-output:/output/ \
        python:3.7-slim cat /output/probability-covid-19.json | python -m json.tool

docker run --rm \
        -v stoicalgorithm-output:/output/ \
        python:3.7-slim cat /output/probability-severe-covid-19.json | python -m json.tool

docker volume rm stoicalgorithm-output

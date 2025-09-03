#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

# make sure that the artifact and scratch folders can be deleted
read -r -p "I will now recursively delete  $SCRIPTPATH/../inference/artifact/*  and   $2/* . Are you sure you want to continue? (type 'yes')? " CONT

if [ "$CONT" != "yes" ]
then
  echo "Aborting."
  exit 0
fi

# make sure the artifact and scratch folders are writable
chmod 777 $SCRIPTPATH/../inference/artifact/
chmod 777 $2/

# Clear the artifact and scratch folder
rm -r $SCRIPTPATH/../inference/artifact/*
rm -r $2/*

# Run the algorithm
MEMORY="128g"

docker run --rm --gpus all \
        --memory $MEMORY --memory-swap $MEMORY \
        --cap-drop ALL --cap-add SYS_NICE --security-opt "no-new-privileges" \
        --network none --shm-size 32g --pids-limit 1024 \
        -v $1:/input/:ro \
        -v $SCRIPTPATH/../inference/artifact/:/output/ \
        -v $2:/scratch/ \
        stoictrain


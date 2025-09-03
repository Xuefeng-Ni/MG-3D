call .\build.bat

docker volume create stoicalgorithm-output

docker run --rm --gpus all^
 --memory=16g --memory-swap=16g^
 --cap-drop=ALL --security-opt="no-new-privileges"^
 --network none --shm-size=128m --pids-limit 256^
 -v %~dp0\test\:/input/^
 -v stoicalgorithm-output:/output/^
 stoicalgorithm

docker run --rm^
 -v stoicalgorithm-output:/output/^
 python:3.7-slim cat /output/probability-covid-19.json | python -m json.tool

docker run --rm^
 -v stoicalgorithm-output:/output/^
 python:3.7-slim cat /output/probability-severe-covid-19.json | python -m json.tool

docker volume rm stoicalgorithm-output

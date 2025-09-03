# STOIC2021 Final Phase Inference Example 

This directory contains an example for generating a Docker container that can do model inference for the Final phase of the STOIC2021 challenge. It is heavily based on [the STOIC2021 submission tutorial for the Qualification phase](https://github.com/luukboulogne/stoic2021-baseline). The key difference between that repo and this folder is that this folder expects the model weights to be located in `./artifact/`. 

## Table of Contents
* [Prerequisites](#prerequisites)
* [Building, testing and exporting your inference container](#buildtestexport)
* [Implementing your own algorithm](#implementing)
* [Tips](#tips)

<a id="prerequisites"></a>
## Prerequisites
Before running the code in this folder, please make sure that you already implemented and executed your training Docker container (see [`../train/README.md`](https://github.com/luukboulogne/stoic2021/tree/master/training)). As a result of executing your training Docker container, the `./artifact/` directory should now contain your trained model.

<a id="buildtestexport"></a>
## Building, testing, and exporting your container 
### Building
To test if your system is set up correctly, you can run `./build.sh` (Linux) or `./build.bat` (Windows), that simply implement this command:

```docker build -t stoicalgorithm .```

Please note that the next step (testing the container) also runs a build, so this step is not necessary if you are certain that everything is set up correctly.

<a name="testing"></a>
### Testing
To test if the docker container works as expected, `test.sh`/`test.bat` will build the container and run it on images provided in the `./test/` folder. It will then check the results (`.json` files produced by your algorithm) against the `.json` files in `./test/`. 

If the tests run successfully, you will see `Tests successfully passed...`.

Notes:
- If you do not have a GPU available on your system, remove the `--gpus all` flag in `test.sh`/`test.bat` to run the test.
- When you implemented your own algorithm using this template, please update the the `.json` files in `./test/` according to the output of your algorithm before running `test.sh`/`test.bat`.


### Exporting
Run `export.sh`/`export.bat` to save the docker image to `./STOICAlgorithm.tar.gz`. This script runs `build.sh`/`build.bat` as well as the following command:
`docker save stoicalgorithm | gzip -c > STOICAlgorithm.tar.gz`

Please note that it is not necessary to include the resulting `STOICAlgorithm.tar.gz` file in your submission. The challenge organizers will build and save a new Docker image using your submission. This new Docker image will be used to test your algorithm on the full test set.

<a id="implementing"></a>
## Implementing your own algorithm
You can implement your own solution by editing the `predict` function in `./process.py`. Any additional imported packages needed for inference should be added to `./requirements.txt`, and any additional files and folders you add should be explicitly copied in the `./Dockerfile`. See `./requirements.txt` and `./Dockerfile` for examples. After implementing your own solution, make sure to [test](#testing) your new algorithm.

Please note that your container will not have access to the internet when executing on grand-challenge.org, so all model weights must be present in your container image. You can test this locally using the `--network=none` option of `docker run`.

<a id="tips"></a>
## Tips
### Running your algorithm on a test folder:
Once you validated that the algorithm works as expected in the [Testing](#testing) step, you might want to simply run the algorithm on the test folder and check the output `.json` files for yourself. If you are on a native Linux system you will need to create a results folder that the docker container can write to as follows (WSL users can skip this step).
   ```
   mkdir ./results
   chmod 777 ./results
   ```
To write the output of the algorithm to the results folder use the following command: 
   ```
   docker run --rm --memory=11g -v ./test:/input/ -v ./results:/output/ STOICAlgorithm
   ```

### Test your algorithm by uploading it to the Qualification phase
The contents in this folder are almost identical to [the tutorial for the Qualification phase](https://github.com/luukboulogne/stoic2021-baseline). You can therefore test your codebase for submission to the Final phase by 
 1. Using a training container based on the template in [`../train`](https://github.com/luukboulogne/stoic2021/tree/master/training) to train your algorithm on [the public training data](https://registry.opendata.aws/stoic2021-training/); 
 2. Generating an inference container using the code in this folder;
 3. Submitting this inference container to the Qualification phase by following the instructions in [the Qualification phase tutorial](https://github.com/luukboulogne/stoic2021-baseline).

# STOIC2021 Final Phase Training Example 

This folder contains an example for generating a Docker container that can train a model on [the public training data](https://registry.opendata.aws/stoic2021-training/) of the STOIC2021 challenge.

## Table of Contents
* [Building, testing and exporting your training container](#buildtestexport)
* [Implementing your own training algorithm](#implementing)

<a id="buildtestexport"></a>
## Building, testing, and exporting your training container 
### Building
To test if your system is set up correctly, you can run `./build.sh` (Linux) or `./build.bat` (Windows), that simply implement this command:

```docker build -t stoictrain .```

Please note that the next step (testing the container) also runs a build, so this step is not necessary if you are certain that everything is set up correctly.

<a name="testing"></a>
### Testing
To test if the docker container for training works as expected, run the `test.sh`/`test.bat` script as shown below. This script contains the commands that will be used by the organizers to train your algorithm. Changing this script will not affect the commands that the organizers will use to train your algorithm. Please do not alter it while preparing your submission.

 `test.sh`/`test.bat` will build the container and train your algorithm with the public training set located in some input directory `/path/to/stoic2021-training/`. In the output directory `../inference/artifact/`, it will produce trained model weights to be used later for generating the inference Docker container.

Your training docker container also has access to some volume at `/path/to/temporary/folder/` that provides 2000 GB of scratch space. This space can be used to store intermediate results such as preprocessed data. **Please do not store such data in the output directory or in another directory inside your docker container**.

To test your training docker container, please make sure that the [STOIC2021 public training set](https://registry.opendata.aws/stoic2021-training/) has been downloaded to `/path/to/stoic2021-training/` and that `/path/to/temporary/folder/` exists. 

**Note: Running the following command will delete the current contents of the output directory `../inference/artifact/` and of the scratch space directory `/path/to/temporary/folder/`.**

You can now run:

```bash test.sh /path/to/stoic2021-training/ /path/to/temporary/folder/```

if you are on Linux, or:

```test.bat \path\to\stoic2021-training\ \path\to\temporary\folder\ ```

if you are on Windows.


If the test runs successfully, you will see you will see `Training completed.`, and your model weights will have been saved to `../inference/artifact/`.

Please note: 
- When the challenge organizers train your algorithm, `/path/to/stoic2021-training/` will be read-only. Please do not try to write to this directory.
- The number of CPUs and GPUs available in the training environment are not specified in the docker run command in `test.sh`/`test.bat`. When training with your submitted codebase, the challenge organizers will use a docker run command with an altered `--gpus` flag so that two 35GB GPUs are used, and with the added flag `--cpus=16`.

<a name="exporting"></a>
### Exporting
Run `export.sh`/`export.bat` to save the Docker image to `./STOICTrain.tar.gz`. This script runs `build.sh`/`build.bat` as well as the following command:
`docker save stoictrain | gzip -c > STOICTrain.tar.gz`

Please note that it is not necessary to include the resulting `STOICTrain.tar.gz` file in your submission. The challenge organizers will build and save a new Docker image using your submission. This new Docker image will be used to train your algorithm on the full (public + private) training set.

<a id="implementing"></a>
## Implementing your own training algorithm
You can implement your own solution by altering the [`do_learning`](https://github.com/luukboulogne/stoic2021/blob/a9916c6a2a8c075300200e0d0c04dfffe93b0b17/training/train.py#L76) function in `./train.py`. See the documentation of this function in `./train.py` for more information.

Any additional imported packages needed for inference should be added to `./requirements.txt`, and any additional files and folders you add should be explicitly copied in the `./Dockerfile`. See `./requirements.txt` and `./Dockerfile` for examples. After implementing your own algorithm, make sure that your training codebase is ready for submission by [testing](#testing) it.

Please note that your container will not have access to the internet when executing on grand-challenge.org, so any pre-trained model weights must be present in your container image. You can test this locally using the `--network=none` option of `docker run`.

Also, please make sure that your training algorithm uses `/path/to/temporary/folder/` to store intermediate results such as preprocessed data. **Please do not store such data in the output directory or in another directory inside your docker container**. Furthermore,`/path/to/stoic2021-training/` will be read-only. Please do not try to write to this directory.

After implementing your own training Docker container, you can implement your inference container. For this, please refer to [`../inference/README.md`](https://github.com/luukboulogne/stoic2021/tree/master/inference).


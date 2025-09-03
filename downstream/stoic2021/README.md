# STOIC2021 Final Phase Submission Template

This codebase contains an example submission for the Final phase of the [STOIC2021 COVID-19 AI Challenge](https://stoic2021.grand-challenge.org/). It implements simple pipelines for training a baseline model and running inference with it. 

The training pipeline trains an [I3D model](https://github.com/hassony2/kinetics_i3d_pytorch) on the [STOIC2021 training data](https://registry.opendata.aws/stoic2021-training/), and produces trained model weights in a specified directory. The inference pipeline uses the trained model weights to predict COVID-19 presence and COVID-19 severity from a CT scan. 

If something in this codebase does not work for you, please do not hesitate to [contact us](mailto:luuk.boulogne@radboudumc.nl) or add a post in the [forum](https://grand-challenge.org/forums/forum/stoic2021-602/). If the problem is related to the code of this repository, please create a new issue on GitHub.

<a id="prerequisites"></a>
## Prerequisites
We recommend using this repository on Linux. If you are using Windows, we recommend installing [Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/install). Please watch the [official tutorial](https://www.youtube.com/watch?v=PdxXlZJiuxA) by Microsoft for installing WSL 2 with GPU support.

* Have [Docker](https://www.docker.com/get-started) installed.
* Have the [STOIC2021 public training set](https://registry.opendata.aws/stoic2021-training/) downloaded to your machine. Instructions for downloading it can be found [here](https://stoic2021.grand-challenge.org/stoic-db/).


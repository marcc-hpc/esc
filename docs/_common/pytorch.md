---
layout: post
title: PyTorch
---

PyTorch is an [open-source machine learning library](https://pytorch.org/get-started/locally/).

## Using PyTorch on *Blue Crab* {#bc}

### Option 1: Use an environment {#bc-env}

The latest version of PyTorch offered on *Blue Crab* is `1.1.0` and can be accessed through an Anaconda module. The following commands are required every time you wish to use it.

``` bash
ml anaconda
conda activate torchvision
```

### Option 2: Use a custom environment

If you wish to install a custom version of PyTorch, you can [make a conda environment](python-environments#conda) with `pytorch::pytorch` in your requirements file. Note that until we upgrade to CUDA 10, you should use `pytorch::pytorch=1.3=*cuda9.2*` to select our highest available CUDA version. Before using the code you should also load the appropriate module with `ml cuda/9.2` to ensure the environment is correctly linked.

The `pytorch` package can be found on [Anaconda cloud](https://anaconda.org/pytorch/pytorch). You are also welcome to install it with `pip` according to their [guide here](https://pytorch.org/get-started/locally/), however you should be mindful of the [caveats on our system](python-environments#pip-caveats).

### Option 3: Legacy versions in containers

*Blue Crab* also offers older versions of PyTorch (`0.4`) in a Singularity container, however this version is somewhat outdated:

``` bash
module spider torch
module load pytorch/0.4.1-gpu-py3
```

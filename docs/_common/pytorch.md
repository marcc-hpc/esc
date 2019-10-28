---
layout: post
title: PyTorch
---

PyTorch is an [open-source machine learning library](https://pytorch.org/get-started/locally/).

## Using PyTorch on *Blue Crab* {#bc}

### Option 1: Use an environment

The latest version of PyTorch offered on *Blue Crab* is `1.1.0` and can be accessed through an Anaconda module. The following commands are required every time you wish to use it.

``` bash
ml anaconda
conda activate torchvision
```

### Option 2: Use a custom environment

If you wish to install a custom version of PyTorch, you can [make a conda environment]() with `pytorch::pytorch` in your requirements file. The package can be found on [Anaconda cloud](https://anaconda.org/pytorch/pytorch). You are also welcome to install it with `pip` according to their [guide here](), however you should be mindful of the [caveats on our system](python-environments#pip-caveats).

### Option 3: Legacy versions in containers

*Blue Crab* also offers older versions of PyTorch (`0.4`) in a Singularity container, however this version is somewhat outdated:

``` bash
module spider torch
module load pytorch/0.4.1-gpu-py3
```

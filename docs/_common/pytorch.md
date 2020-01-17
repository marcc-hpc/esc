---
layout: post
title: PyTorch
---

PyTorch is an [open-source machine learning library](https://pytorch.org/get-started/locally/).

## Using PyTorch on *Blue Crab* {#bc}

### Option 1: Use an environment {#bc-env}

Two versions of PyTorch are available on *Blue Crab* without using a custom environment. To use either `1.1.0` or `1.4.0`, you can load the system-installed Anaconda module and then select a shared pre-installed environment. We are preparing to upgrade to CUDA 10 in the near future. Until then, our software is compatible with CUDA 9.2. To load PyTorch `1.4.0`, use the following commands.

``` bash
ml anaconda
ml cuda/9.2
conda activate pytorch-1.4
```

To load PyTorch `1.1.0`, use the following commands.

``` bash
ml anaconda
ml cuda/9.2
conda activate torchvision
```

Other shared environments provided by the Anaconda module (`ml anaconda`) can be listed with `conda env list`. Users can confirm that their GPU is available with the following code.

```
# interactive session on high-availability GPU debug nodes
interact -p debug -g 1 -c 6 -t 20
ml anaconda
conda activate pytorch-1.4
python -c "import torch; torch.cuda.is_available()"
# True
```

If you require additional Python packages alongside PyTorch, you are welcome to install them to `~/.local` with `pip install --user`, however please keep in mind that these versions may cause conflicts if you later switch Python versions (the PyTorch modules above use Python 3.7). For greater control over your software versions, you should build a custom environment using the instructions below.

### Option 2: Use a custom environment

If you wish to install a custom version of PyTorch, you can [make a conda environment](python-environments#conda) with `pytorch::pytorch` in your requirements file. Note that until we upgrade to CUDA 10, you should use `pytorch::pytorch=1.3=*cuda9.2*` to select our highest available CUDA version. Before using the code you should also load the appropriate module with `ml cuda/9.2` to ensure the environment is correctly linked.

The `pytorch` package can be found on [Anaconda cloud](https://anaconda.org/pytorch/pytorch). You are also welcome to install it with `pip` according to their [guide here](https://pytorch.org/get-started/locally/), however you should be mindful of the [caveats on our system](python-environments#pip-caveats).

### Option 3: Legacy versions in containers

*Blue Crab* also offers older versions of PyTorch (`0.4`) in a Singularity container, however this version is somewhat outdated:

``` bash
module spider torch
module load pytorch/0.4.1-gpu-py3
```

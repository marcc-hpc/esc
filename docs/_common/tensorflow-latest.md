---
layout: post
title: "TensorFlow: latest"

---

During the outage in mid-March we have upgraded the hardware in our GPU partitions (namely `gpuk80`) to `CUDA` driver `440.33.01` which supports TensorFlow up to version `2.2`. The following guide will show you how to install them with a [virtual environment](#venv) or [Anaconda](#conda).

The following guide largely repeats the instructions in our [software environments](python-environments) guide. The only specific instructions we have added are:

1. Use the `cuda/10.1` software module.
2. Install TensorFlow 2.2.0 (specifically `2.2.0rc0`) with `pip` inside your environment.

Read below for a step-by-step walkthrough.

### Option 1: Virtual Environments {#venv}

TensorFlow typically requires the penultimate version of Python. To use a virtual environment to install TensorFlow 2 you must load a Python 3 module and select a location for a virtual environment. We strongly recommend using the new `~/code` location inside your home directory. Do not install programs on our Lustre filesystem (`~/scratch` and `~/work`). The following steps will build the virtual environment and install TensorFlow inside. Lines prefixed with the hash symbol are comments for your reference.

~~~
# start an interactive session on a GPU node
interact -p debug -c 6 -t 60 -g 1
# navigate to your ~/code directory or another suitable location
cd ~/code
# load the Python and CUDA modules
ml python/3.7.7
ml cuda/10.1
# build the environment in a folder called ./venv
python -m venv ./venv
# activate the environment
source ./venv/bin/activate
# install tensorflow 2.2
pip install tensorflow==2.2.0rc0
# confirm that it is installed
python -c 'import tensorflow'
~~~

After installing the environment you will access it with the following commands. Note that the path to your environment may be different it if you installed it elsewhere.

~~~
ml cuda/10.1
source ~/code/venv/bin/activate
~~~

Note that we recommend installing TensorFlow 2.2.0 for use with CUDA 10.1. 

### Option 2: Anaconda {#conda}

[Anaconda](https://anaconda.org/) allows you to carefully control the exact version of Python and install additional supporting packages.

Select a location, ideally in `~/code` or on the `~/data` filesystem (but not our Lustre filesystem at `~/scratch` and `~/work`). Write the following `reqs.yaml` file.

~~~
dependencies:
  - python==3.7
  - pip
  - pip:
    - tensorflow==2.2.0rc0
~~~

Install the environment to your chosen location with the following commands.

~~~
# start an interactive session on a GPU node
interact -p debug -c 6 -t 60 -g 1
# navigate to your ~/code directory or another suitable location
cd ~/code
# load the Anaconda and CUDA modules
ml anaconda/2019.03
ml cuda/10.1
# build the environment at your chosen location 
conda env update --file ./reqs.yaml -p ./env
# activate the environment
conda activate ./env
# confirm that it works
python -c 'import tensorflow'
~~~

After installing this environment, you can access the environment with the following commands.

~~~
ml anaconda/2019.03
ml cuda/10.2
conda activate ~/code/env
~~~

Note that the paths above may be different if you install your environment to a different location.

### Pending updates

In the future we will make TensorFlow automatically available in a module format on our [new software modules](software-modules#new). Watch this space for updates.

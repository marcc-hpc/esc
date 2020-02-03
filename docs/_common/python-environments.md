---
layout: post
title: Python environments
# todo: link to portal tools e.g. Jupyter
# todo: This method uses `conda env` however it is also possible to use `conda` directly to install [packages in sequence](#conda-sequence).
# todo: complete the list below
---

This page explains the use of Python environments. Users who are interested in using `R` may also find the desciption of [Anaconda](https://www.anaconda.com/) useful.

*This page is under construction. More options coming soon. Check out [this site](https://marcc-hpc.github.io/tutorials/shortcourse_python.html) in the meantime.*

## Options for controlling Python environments {#options}
We encourage all users to carefully select the best environment option for their software targets.

1. Use the **default modules**. If your desired code is available in our software modules (`ml av`) or on our [new software tree](new-stack) then this option is the best option because it provides the code you need with very little extra configuration.
2. Install a **virtual environment**. For codes which are not available in the default modules, the quickest solution is to use a python virtual environment.
3. [Build a **`conda` environment**](#conda). The [Anaconda cloud](https://anaconda.org/anaconda/python) provides a larger set of both Python packages as well as executables which you might otherwise install with your operating system package manager. This option provides the most customization.

## A word of caution: user-installed codes

If you use `pip install --user` to install software, it often preempts the methods described below. This method will install packages to `~/.local` however the precise path depends on which version of Python you are using when you install the code. Because this method is opaque, we recommend using virtual environments instead.

## A. Python virtual environments {#venv}

If you cannot find the right software on our module tree (`ml av`) or our [new software stack](new-stack), and the code is distributed on [`PyPI`](https://pypi.org/), then you can easily install it in a virtual environment.

1. Select a python version using our modules system. 
2. Select a location to install your environment on the `~/data` mount. Do not use the Lustre filesystem (`~/scratch` and `~/work`) for this.
3. Build the environment with `python -m venv ./path/to/env`.
4. Activate the environment with `source ./path/to/env/bin/activate`. You will have to use this command whenever you wish to use the environment.
5. Inside the environment you can install packages normally, without the use of the `--user` flag, with `pip`. For example, you can install `numpy` with: `pip install numpy`. 

Virtual environments can be a useful tool for reproducing your workflow on other machines using the `pip freeze` command, which lists all of the packages you have already installed.

## B. Custom conda environments {#conda}

Note that this is the best solution for users who need to control their Python (or `R`) version, install packages with `pip`, use interactive portal tools, or install `conda` packages from [Anaconda Cloud](https://anaconda.org/anaconda/python). The use of `conda env` ensures that you can generate your environment from an easily-portable text file. This also resolves version dependencies all at once, so that you don't paint yourself into a corner. (Users who experience perl version issues on *Blue Crab* should consult [this guide](perl-version-issue).)

### 1. Find a useful location to make an environment

Typically, the `conda create` command will make an environment in a hidden folder in your home directory, at `~/.conda`. These so-called "named" environments are suboptimal because you might forget about them and fill up your quota. It's much better to specify an absolute path to the environment so you always know where it is.

On *Blue Crab* we recommend that all users install to `~/` (your home directory, be mindful of the quota) or your shared group directory on our ZFS filesystem at `~/data`. The use of our Lustre system at `~/work` and `~/scratch` is discouraged because these installations create many small files and this filesystem is not optimal for repeatedly executing binaries.

Once you find a spot to install the environment, you can continue.

### 2. Prepare a requirements file

The requirements file consists of a `dependencies` list which enumerates the `conda` packages that you might typically install with `conda install`. You can specify a channel with the special syntax below (`::`) when packages are not available on the primary channel. We have also included `pip`, so that `conda` can also manage packages from [PyPI](https://pypi.org).

~~~
dependencies:
  - python=3.7
  - matplotlib
  - scipy
  - numpy
  - nb_conda_kernels
  - au-eoed::gnu-parallel
  - h5py
  - pip
  - pip:
    - sphinx
~~~

Save this as a text file called `reqs.yaml`. The file respects the [YAML](https://yaml.org/) format. Note that the use of `nb_conda_kernels` is necessary to use this environment in Jupyter on *Blue Crab*. Users who wish to later install their packages with `pip` should include it on the list. This will allow you to run `pip install` to add a package to this environment and avoid polluting your `~/.local` folder when using, `pip install --user`, which is a typical alternative.

### 3. Install the environment

We recommend choosing a useful name for your environment. This will help distingish your environment from others, in case you later use portal tools like Jupyter. Install Anaconda, or if you are using *Blue Crab*, load the anaconda module. Recall that `ml` is short for `module load` when using [Lmod](https://lmod.readthedocs.io/en/latest/).

~~~
ml anaconda
~~~

After selecting a name (`my_plot_env`), install the environment. If the environment installation is complex, you may wish to reserve a compute node.

~~~
conda env update --file reqs.yaml -p ./my_plot_env
~~~

The command above *can be executed over and over again as you add new packages to your environment*. We recommend that 

After the environment is installed, you can use it in future terminal sessions or scripts using:

~~~
ml anaconda
conda activate ./path/to/my_plot_env
~~~

### Benefits

The method above offers the following benefits:

1. You can repeatedly update `reqs.yaml` to add new packages, and then use the `conda env update` command above to add them to the environment. This helps to prevent version conflicts and makes your work more reproducible.
2. As long as you include `pip`, you can add packages like `sphinx` in the example requirements file. The `conda` program can manage packages from its own repositories as well as the python package index ([PyPI](https://pypi.org)).
3. The use of `nb_conda_kernels` will send a signal to Jupyter so you can use this environment inside a notebook.
4. Oftentimes conda packages are faster. You may notice that `numpy` uses the Intel MKL library, for example, which provides a large speedup for linear algebra calculations on some platforms.

### Alternative: installing a conda environment sequentially {#conda_seq}

While we strongly recommend that users maintain a requirements (`reqs.yaml`) and make use of the `conda env update` command described above, it is also possible to install individual packages into your environment. We recommend choosing a location on `~/data` (using the path flag, `-p`) to store your environment.

~~~
ml anaconda
# cd to a location on ~/data
conda create -p ./myenv
conda activate ./myenv
# install pip
conda install pip
# after you install pip, you should use pip directly (without --user)
pip install matplotlib
# install a conda package from a specific channel
conda install -c au-eoed gnu-parallel
~~~

Note that this method can often cause package upgrades and downgrades in order to resolve a set of mutual dependencies. For this reason, we recommend using a requirements file instead, so that your environment can be reproducibly generated from a single set of requested packages.

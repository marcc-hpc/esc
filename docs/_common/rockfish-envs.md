---
layout: post
title: Rockfish environments
---

The following notes summarize the software environments on the next-generation *Rockfish* cluster at Johns Hopkins University.

## The Bird's Eye View

The *Rockfish* cluster uses [Spack](https://spack.readthedocs.io) as a package manager along with [Lmod](https://lmod.readthedocs.io/en/latest/) to provide an environment modules system. We employ a hierarchical modules system that enforces a single compiler and MPI (message passing interface) combination. We also provide Python and R packages in the modules system for common or highly complex dependencies in these languages. Users are encouraged to use these modules, first before trying virtual environments, Anaconda, or Singularity. Strategy advice for this process is provided below, and we generally outline five scenarios.

1. [Use a compiler and MPI implementation to compile your own code.](#custom)
2. [Use the software modules.]({#modules})
3. [Use Python or R packages in the modules system.]({#spackpack})
4. [Install your own packages in a virtual environment.]({#venv})
5. [Install your own packages in an Anaconda environment.]({#conda})
6. [Run your software from a container.]({#container})

We will cover each of these scenarios in the following guide.

## What makes HPC unique?

High-performance computing (HPC) must include hundreds (or more) users to effectively leverage large investments in a physical machine. This means that many people must share a single large computing cluster. The so-called multi-tenant nature of these machines means that the engineers and administrators must deliver software for *everybody* all at once. This is accomplished with two tools:

1. A package manager that can provide *multiple* versions of many pieces of software. We use [Spack](https://spack.readthedocs.io) to accomplish this.
2. A modules system to keep the software separate and accessible. We use [Lmod](https://lmod.readthedocs.io/en/latest/) for this.

For the average user, this means that some amount of effort is required to find or install or adapt the code you want to use on our system. In practice, this means that you probably cannot simply follow installation instructions for custom codes unless they are specifically designed to be installed on systems where you do not have superuser or root permissions. If the documentation for your desired software has the words `sudo`, or `apt-get` or `yum install` in the notes, then you probably cannot install it on our system.

Instead, you can often find pre-installed equivalents using `module spider`. Since we offer many similar pieces of software, you may need to select the version carefully. In the ideal case, you can translate the "install this on my workstation" instructions into "install this on a cluster" by searching for the right modules, and reading the documentation for the software dependencies you require. Our staff can help. In the remainder of the guide we will outline the five most common ways this is accomplished.

## Scenario 1: custom code {#custom}

If you only need a compiler and MPI implementation, then you can compile your code from source. We offer GCC and Intel compilers, along with OpenMPI and IntelMPI implementations. This is the high-performance computing bread and butter: you need a compiler to build your software and MPI middleware to tell it how to communicate over our massively parallel system if it needs multiple nodes. Multi-processor applications may not need an MPI if they must only run in parallel on a single node, however they may still require parallel frameworks offered by the compiler and other tools in our software stack.

## Scenario 2: software modules {#modules}

The most common way that you will use an HPC cluster is by taking advantage of pre-compiled software provided in the *modules* system. The modules allow you to select any number of individual pieces of software for use in combination. You can load and unload software modules to ensure that your programs can find the correct compilers and libraries. This all starts with commands like this:

```
# see the avilable modules
module avail
# load a compiler and MPI
module load gcc/9.3.0 openmpi/3.1.6
# list your modules
module list
```

We enforce a single compiler and MPI according to guidance [summarized in the Lmod documentation](https://lmod.readthedocs.io/en/latest/080_hierarchy.html). When you log on, your default modules will include GCC and OpenMPI. These defaults might change at a later date. When you select a compiler and MPI combination, Lmod *will *reveal packages that have been compiled under this compiler and MPI.* If you are looking for software, and you cannot see it on the `module avail` listing, you must use `module spider <name>` to search for the package. The Lmod system will tell you how to load the package. Some packages may only be revealed after you load the correct compiler and MPI. Users should consult the [Lmod documentation](https://lmod.readthedocs.io/en/latest/) to learn how to search for software.

*Users are encouraged to think critically about the right compiler for their workflow, since it may make a very larger performance difference.*

## Scenario 3: Interpreted languages {#spackpack}

While most HPC software is compiled into a binary from source, a large number of users use interpreted languages, particularly Python and R, which are the two most popular on academic systems. Both have extremely broad libraries of code compiled in other languages, high performance benefits, and broad adoption in many academic domains. 

Since so many users require such a diverse set of Python and R libraries, we have included many of the most popular libraries as modules. This is fairly uncommon. Most Python and R users will first install the programming language and then install a custom set of "libraries" or "modules" alongside the language itself. Since this process has many combinations and common use-cases, we have used [Spack](https://spack.readthedocs.io) to install them ahead of time.

We do not load Python or R by default, so users who first log on will only have access the system Python. We recommend loading Python from the modules system using `module load python`. When you load the module, the `ml av` command will reveal Python packages you would otherwise have to install with another package manager. These have prefixes (`py-<name>`) and can be loaded as modules. For example, rather than installing [`numpy`](https://numpy.org/) with `python -m pip` to your local directory (which is hidden at `~/.local`), you can use the system-compiled copy by running `module load numpy` instead. 

This has two benefits. Whenever you use the modules system you are guaranteed to avoid incompatible versions. The modules system also allows you to easily keep track of your software (use `module list`) which helps you build reproducible workflows.

When you switch compilers, Lmod will swap out the available software so that all of the Python or R packages are still compatible. This is even true for libraries in a language like Python.

```
$ ml python/3.8.6 py-numpy
$ ml python/3.7.9

Due to MODULEPATH changes, the following have been reloaded:
  1) py-numpy/1.18.5     2) py-setuptools/50.1.0

The following have been reloaded with a version change:
  1) python/3.8.6 => python/3.7.9
```

Not every version of Python has the same set of packages. The modules system will tell you when switching one packages causes another package to become "inactive" since it is not compatible. You will notice that loading Python and R will reveal large sets of packages under the corresponding header in the `module avail` listing. You should try loading these packages with `module load` before using Python or R, since this can help you avoid installing your own copy with `python -m pip install` (for Python) or `install.packages` (for R). 

## Scenario 4: Python virtual environments {#venv}

In the event that you need a package which is *not* listed on `module avail` after you load `module load python`, you should consider a virtual environment. A virtual environment can install any code you find on the Python Package Index at [pypi.org](https://pypi.org). While users commonly use the `pip` or `pip3` command to install these packages, we strongly recommend that you use `python` itself to invoke the installer. On RockFish, you should use the following procedure.

~~~
# load a python module along with pip and setuptools
ml python/3.8.6 py-pip py-setuptools
# select a path in your home directory and make an environment
python -m venv ./myenv
# activate the environment
source ./myenv/bin/activate
# install packages from PyPI
pip install psutil
# test
python -c 'import psutils'
~~~

As an alternative, you can skip the virtual environment and install directly to a single, default environment which is associated with a particular Python module. This will install packages into a hidden folder at `~/.local` which is specific to your Python version.

~~~
# load a python module along with pip and setuptools
ml python/3.8.6 py-pip py-setuptools
# install to ~/.local/
python -m pip install --user psutil
# test
python -c 'import psutil'
~~~

Since this folder is hidden, and you must be careful to select the right Python version (since the packages cannot be mixed up), we recommend the virtual environment method instead. If you switch to another python version (e.g. `ml python/3.7.9`) then you will not be able to load the `psutil` library you installed above.

## Scenario 5: Anaconda {#anaconda}

The [Anaconda Cloud](https://anaconda.org/) offers an alternative package manager which provides access packages across several languages including Python and R. It also allows you to install from the the Python Package Index. This section will explain the best practices for using Anaconda on RockFish.

**Anaconda as a module.** Before using `conda` on RockFish it is important to understand a key difference compared to your workstation. While Anaconda can install many versions of Python, it is designed to provide a self-contained environment based on a single version. Typically, users install the program, and then run a `conda init` command, and after that point, their `~/.bashrc` will automatically load the `(base)` Anaconda environment. We avoid this strategy because it restricts access to the Lmod system and it can be confusing. Instead, we offer a customized anaconda module which can be used without modifying your `~/.bashrc`. You should ***not need to edit your*** `~/.bashrc` to use `conda` on RockFish. 

To use the `conda` command, simply run `ml anaconda`. To install an environment, we recommend preparing an environment specification. For example, this environment installs [Jupyter](https://jupyter.org/). For the reasons listed above, you cannot load the `anaconda` module at the same time as the `python` module.

~~~
dependencies:
  - python==3.8.3
  - jupyter
  - nb_conda_kernels
  - pip
  - pip:
    - pyyaml
~~~

Save this file as `reqs.yaml` and keep it with your lab notebook for later. To install or update your environment from the contents of this file, use the following command. Note that we have selected an explicit path, instead of a name, to install the environment. Named environments are installed to a hidden folder (`~/.conda`) while explicit paths allow us to see our environment.

~~~
ml anaconda
conda update env --file reqs.yaml -p ~/env-project-01
~~~

To use this environment, run the following commands. These should be included in any scripts or workflows that require this environment.

~~~
ml anaconda
conda activate ~/env-project-01
~~~

Note that you can use any path for your environment, but we recommend your home directory because it is optimized for code and backed up.

For reproducibility purposes, you should keep a copy of `reqs.yaml` for your records, and ideally avoid the use of `conda install`. Instead, update the `reqs.yaml` file and run the `conda update env` command above, to add packages. You can also remove packages according to the [conda docs](https://docs.conda.io/projects/conda/en/latest/commands/remove.html). In addition to the minimal instructions in `reqs.yaml` you should also run the following command to save a more detailed list.

~~~
conda env export -p ~/env > env_freeze.yaml
~~~

The resulting `env_freeze.yaml` file includes all of the required packages and their version numbers. It can be used to exactly reproduce your environment on another machine. Oftentimes we change our environments over time. The minimal `reqs.yaml` is somewhat future-proof if your packages have newer versions available, while the "frozen" version is a failsafe for reproducing your environment later on.

## Scenario 6: Containers {#containers}

If you cannot deploy your desired software using the methods above, you can use [Singularity](https://sylabs.io/singularity/) to deploy containers (including [Docker](https://www.docker.com/) containers). These provide the maximum level of customization.

*Note* that Singularity is not yet installed on RockFish. Documentation is pending installation.
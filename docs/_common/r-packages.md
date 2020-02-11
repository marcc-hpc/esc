---
layout: post
title: R Packages
# finish this page
# link to a privatemodules section in case 3 when users compile code manually
# quantify the rcpp cost and emphasize it
---

This page provides a guide for installing packages in `R` on *Blue Crab*. Most users accomplish this with `install.packages`, however the rich ecosystem of `R` packages requires some additional effort to install on HPC resources.

*If you were referred to this site by a member of the MARCC support staff, please read through all of cases below to determine which strategy is best for your software.* Correspondence with <code><nobr><a href="marcc-help@marcc.jhu.edu">marcc-help@marcc.jhu.edu</a></nobr></code> should include a complete description of your strategy.

# Cases

The following list describes the methods for installing R packages on *Blue Crab*.

1. [**Software Modules**](software-modules#new) provide R packages directly, as long as you are using our [new software modules](software-modules#new).

## Case 1: Software Modules

Many common packages have been added to our [new software modules](software-modules#new) which are accessible via [Lmod](https://lmod.readthedocs.io/en/latest/) and the `ml av` command. You can switch to the new modules by using `ml stack`, however we recommend that all uses [consult the guide when choosing this option](software-modules#new). 

This option is best for users who need the common packages installed in the `ml stack` software modules. We have built these packages to sidestep the typical `install.packages` method which is time-consuming and often depletes the free space in your home directories. Therefore, if you can find your software with this method, it will save you time and space.

By way of example, imagine that you wish to install `Rcpp`. Typically you would run `install.packages('rcpp')` however this requires more than a few minutes of time and creates many extra files. Instead, use the following method.

~~~
$ ml stack
$ ml av
$ ml r/3.6.1
$ ml r-rcpp
$ R
> library('rcpp')
~~~

This provides the `Rcpp` package with minimal effort. We over dozens of additional R packages, all of which are kindly provided by [Spack](https://spack.readthedocs.io/en/latest/), on our [new software modules](software-modules#new).

## Case 2: Direct installation

The example above, in which we used the `Rcpp` package installed on our software modules system, is the best-case scenario because it provides software without a custom installation. If you cannot find your software on `ml stack`, then you can install it yourself with `R` directly, using `install.packages`. Before doing so, please consult the documentation for your target. 

Many R packages require additional packages to be *already installed* on your system. The documentation almost always recommends that you install it yourself using e.g. `sudo apt-get install`. Since MARCC provides a shared HPC cluster, this is forbidden, hence you must consult the [software modules](software-modules) to find the right packages. 

For example, imagine that you require a package which depends on the `rJava` package. If you try to install this directly, you will receive an error message that says: "Unable to run a simple JNI program." To overcome this message, you should consult our modules system with `ml av` and load the Java module first. *The modules system therefore takes the place of a typical package manager.* Your installation procedure would then proceed as follows.

~~~
$ ml R/3.6.1
$ ml java
$ R
> install.packages('rJava')
~~~

We encourage all users to start an interactive session with `interact` to install these packages for performance and network connectivity reasons.

## Case 3: Extra dependencies required

If you cannot install your software according to the two cases above, you will need to manage your extra dependencies in one of two ways, by either compiling the code from source, or by using Anaconda. Since both of these options require extra work, we *strongly recommend checking for your software on our software tree first*.

If you are able to compile your supporting packages, then you typically only need to add its path to your environment variables. For the remainder of this section we will explain how to install the `pdftools` package in R because it provides a useful example of an R package with extensive software requirements which are *not* available on our software tree. To overcome this problem, we will instal a *completely independent* copy of R and the supporting packages. We will follow the method for [building Anaconda environments](python-environments#conda). 

First, find a location on your `~/data` directory to install a new environment. Do not use the Lustre filesystem (`~/scratch` and `~/work`) for this. We will use an absolute path to build our software environment. To build the environment we will first create a "requirements" file called `reqs.yaml` with the following contents.

~~~
dependencies:
  - conda-forge::r-pdftools
~~~

This file uses the standard [`YAML`](https://yaml.org/) syntax. Once you prepare this file and choose a location, you can build the environment. You may wish to use `interact` to build this on a compute node.

~~~
$ ml anaconda
$ conda env update --file ./path/to/reqs.yaml -p ./path/to/env
$ conda activate ./path/to/env
$ R 
> library('pdftools')
~~~

By requesting the `r-pdftools` package from [Anaconda cloud], we have also installed an independent R distribution which is entirely separate from the software modules. Whenever you wish to use this environment, you will have to run the following commands. 

~~~
$ ml anaconda
$ conda activate ./path/to/env
~~~

Do not modify your `~/.bashrc` and do not use `source activate` to access this environment. Instead you must always use `conda activate` along with the absolute path, as well as `conda deactivate` to leave the environment. See the [guide to Anaconda environments](python-environments#conda) for more details. In short, the file above takes the place of a `conda install` command which installs the `pdftools` packages from the `conda-forge` channel.

You are welcome to continue adding additional packages based on Python, R, or any package available on [Anaconda cloud](https://anaconda.org/anaconda/python) using the method described above. This method improves reproducibility because it allows you to reconstitute your environment on new hardware from your requirements file.
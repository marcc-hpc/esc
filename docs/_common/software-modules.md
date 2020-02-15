---
layout: post
title: Software modules
# add examples for Python and R packages
# add tensorflow for stack
# add a complete list for stack
# explain the nested modules on the new tree are there for reference
---

*Blue Crab* hosts two extensive sets of software modules accessible by the [`Lmod` environment modules system](https://lmod.readthedocs.io/en/latest/). Most users will start with the default modules which we call the "[**original software modules**](#original)" and can be viewed with `module avail` after you log on. We also offer "new software modules" which provide additional packages installed with the [Spack](https://spack.readthedocs.io/en/latest/) build system. The "[**new software modules**](#new)" offer the newest set of compilers and package options. More importantly, they also provide a number of supporting Python and R packages which can save users the effort of configuring these packages on their own.

## The "Original" Software Modules {#original}

*Blue Crab* provides a large number of software modules using the [`Lmod` environment modules system](https://lmod.readthedocs.io/en/latest/). Users can view the modules using the `module avail` or `ml av` command. You can load a module by using the `module load` command. For example, to use Python 3.7, you can run `module load python/3.7` or `ml 3.7` for short. 

*Not all modules are visible at all times.* We show software which has been compiled with a single compiler (either Intel or GCC) and a single MPI implementation (OpenMPI or IntelMPI). We do not allow mixing of compilers. This means that switching from the default Intel compiler to GCC will reveal a new set of modules based on the GCC compiler. The upshot is that you must use a special command to search for modules which you cannot see. Use a command like `module spider lammps` to search the entire tree for a piece of software. It will tell you if you need to switch compilers to load it. The following example demonstrates this method.

~~~
$ ml list
Currently Loaded Modules:
  1) centos7/current   2) intel/18.0   3) openmpi/3.1   4) MARCC/summer-2018

$ ml spider lammps
ml spider lammps
----------------------------------------
  lammps:
----------------------------------------
     Versions:
        lammps/2016-ICMS
        lammps/20180822-gpu
        lammps/20180822
        lammps/20190208
        lammps/20190329
        lammps/20190514
----------------------------------------
  For detailed information about a 
  specific "lammps" module (including 
  how to load the modules) use the 
  module's full name.
  For example:

     $ module spider lammps/20180822
----------------------------------------

$ module spider lammps/20190514
----------------------------------------
  lammps: lammps/20190514
----------------------------------------

  You will need to load all module(s) 
  on any one of the lines below before 
  the "lammps/20190514" module is 
  available to load.

      gcc/5.5.0  openmpi/3.1

$ module load gcc/5.5.0

Lmod is automatically replacing "intel/18.0" with "gcc/5.5.0".

Due to MODULEPATH changes, the following have been reloaded:
  1) openmpi/3.1

$ module load openmpi/3.1

$ module load lammps/20190514

$ which lmp
/software/apps/lammps/20190514/gcc/5.5/openmpi/3.1/bin/lmp

$ ml
Currently Loaded Modules:
  1) centos7/current     3) gcc/5.5.0     5) cuda/9.2   7) fftw3/3.3.8
  2) MARCC/summer-2018   4) openmpi/3.1   6) gsl/2.5    8) lammps/20190514
~~~

In the workflow above we have listed our modules, used `module spider` to search for instructions to load the latest available [LAMMPS](https://lammps.sandia.gov/) package, and then used the `module load` commands to access the right software. We call this system a *hierarchical* modules system because we enforce the use of a single compiler and MPI implementation to constrain the set of available codes. This reduces compatibility problems.

Users with further questions about the modules system should [read the documentation for Lmod](https://lmod.readthedocs.io/en/latest/) for more details. 

## The "New" Software Modules {#new}

The latest software builds on *Blue Crab* are now available through an alternative modules tree. These have been delivered with the [Spack](https://spack.readthedocs.io/en/latest/) package management tool which has been developed by a large community of very generous programmers who help to make a more uniform and comprehensive set of software available to scientists.

To use the "new" software modules, run `ml stack/0.3` or `ml stack/0.4` for the pre-release version.

### History

The original software modules were built after our August 2018 upgrade. In 2019 we started offering new software inside of a separate set of Lmod software modules using [Spack](https://spack.readthedocs.io/en/latest/). New and complex software requests will be added to this module tree in the future. 

The new stack can be accessed by running `ml stack` which loads a special module that replaces all of the available modules. You must run this command *once per terminal session* unless you add it to your `~/.bashrc` file. After you load the `stack` module, try `module avail` to see the new additions. As with the [original software modules](#original), the modules are hierachical, which means that loading a different compiler or MPI implementation will change the available list. The default compiler is `gcc/7.4.0` and the default MPI is `openmpi/3.1.5` on the new software modules. *To return to the original modules,* use `ml -stack` and your modules will be reset.

Note that there may be some lingering issues with module collections, including the default collection. Please consult our [support team](mailto:marcc-help@marcc.jhu.edu) if you encounter one of these issues.

### Releases

We maintain both a *current* and *pre-release* version of the `stack` module. We recommend taking note of which one you are using. After you load the default with `ml stack`, use `ml` to see which version you are using. The current release is `ml stack/0.3`. You should use this in your SLURM scripts to maintain a stable environment. We may upgrade or remove packages over time. We also offer a pre-release version with new software available if you use `ml spider` to list the versions.

### Extra compilers

The primary benefit to the new stack is that we can easily supply almost any package [provided by Spack](https://spack.readthedocs.io/en/latest/package_list.html). This helps improve the number and combination of supporting packages, including compilers and MPI implementations. As of 2019 this includes the following items which cannot be found on the default modules list:

- `gcc/6.3.0`
- `gcc/7.4.0`
- `gcc/9.2.0`
- `llvm/8.0.0`

The entire list includes many more complex packages. We encourage all users to consult the list when they need new software. 

### Returning to the original modules

You cannot unload the `stack` module (if you try, you will see a warning message). This differs from the standard [Lmod](https://lmod.readthedocs.io/en/latest/) method in order to accomodate our use of two entirely distinct sets of modules. Instead, use `ml original` to return to the default modules system if the new stack does not interest you. We provide hints when you load the `stack` module to remind you of the right command. If you want to permanently use the new modules, load `stack` and then save your default modules with `ml save` which will save your current modules as the default for next time

### More efficient R packages {#stack_R}

The R package provided by `r/3.6.1` on the new software stack also provides packages via modules. Typically a user who wants to use the `devtools` package will install a local copy using `install.packages` inside an R session. This is both time-consuming and wastes disk space. Users who wish to automatically load R packages from the list (e.g. `r-devtools` and `r-ggplot`) can use these commands.

```
ml r/3.6.1
ml r-devtools
ml r-ggplot2
R
> library(ggplot2)
```

Please let our [support team](mailto:marcc-help@marcc.jhu.edu) know if you are interested in other R packages [provided by Spack](https://spack.readthedocs.io/en/latest/package_list.html).

---
layout: post
title: The new software stack
---

## Default software modules

*Blue Crab* currently offers a large set of software compiled with the Intel and GNU compilers. You can list the available modules using `module avail` however loading a different compiler or MPI will change the availables codes because our system uses [Lmod hierarchical modules](https://lmod.readthedocs.io/en/latest/080_hierarchy.html). As of late 2019, we are still adding new software modules.

## Adding new software

In 2019 we have started offering new software inside of a separate set of Lmod software modules which we call "the new stack" to distinguish it from the defaults described above. New and complex software requests will be added to this module tree. The entire tree is supported by the excellent **[Spack](https://spack.readthedocs.io/en/latest/package_list.html)** tool which allows us to provide many more packages to our users.

The new stack can be accessed by running `ml stack` which loads a special module that replaces all of the available modules. You must run this command once per terminal session, including those within SLURM jobs, to use this software. After you load the "stack" module, try `module avail` to see the new additions. As with the original stack, the modules are hierachical, which means that loading a different compiler or MPI implementation will change the available list. The default compiler is `gcc/7.4.0` and the default MPI is `openmpi/3.1.4` on the new stack.

You cannot unload the module to return to the original tree. Instead, use `ml original` to return to the default modules system if the new stack does not interest you. 

The solution described above is *temporary* until we can migrate our entire software stack to the new system. This may never happen if the new modules are not popular enough, in which case users will simply have to opt-in to the new module system with `ml stack`. This is a reasonable cost for the extra software it provides. As always, users are welcome to inspect the modules with `module show` to read the paths for themselves.

### Extra compilers

The primary benefit to the new stack is that we can easily supply almost any package [provided by Spack](https://spack.readthedocs.io/en/latest/package_list.html). This helps improve the number and combination of supporting packages, including compilers and MPI implementations. As of 2019 this includes the following items which cannot be found on the default modules list:

- `gcc/6.3.0`
- `gcc/7.4.0`
- `gcc/9.2.0`
- `llvm/8.0.0`

The entire list includes many more complex packages. We encourage all users to consult the list when they need new software.

### Open issue: module conflicts

Some users have reported that loading the new stack with `ml stack` followed by an interactive slurm session may cause a conflict. This is caused by our use of an *ad hoc* solution for supplying *two separate* module trees. This may cause an error that looks like this: `The following module(s) are unknown: "stack"`. If you see this error, then you already have the new stack loaded, or you need to run `ml original` first. This issue is pending resolution as of 12 November 2019.

## More efficient R packages {#stack_R}

The R package provided by `r/3.6.1` on the new software stack also provides packages via modules. Typically a user who wants to use the `devtools` package will install a local copy using `install.packages` inside an R session. This is both time-consuming and wastes disk space. Users who wish to automatically load R packages from the list (e.g. `r-devtools` and `r-ggplot`) can use these commands.

```
ml r/3.6.1
ml r-devtools
ml r-ggplot2
R
> library(ggplot2)
```

Please let our support team know if you are interested in other R packages [provided by Spack](https://spack.readthedocs.io/en/latest/package_list.html).

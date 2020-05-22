---
layout: post
title: "Managing R packages: a case study"
---

In this guide, we will outline a brief case study in which a user must install a set of `R` packages on *Blue Crab*. We will focus on several packages: `sf`, `gdal`, `proj`, and `geos`.


Broadly speaking, there are three methods for accomplishing this:

1. Use the [new software modules](r-packages#new-modules).
2. Direct installation [within R](r-packages#direct).
3. Use another package manager, [namely Anaconda](r-packages#r-via-conda).

For this guide, we will assume that options 1 and 3 are not optimal. Many users prefer to stick with the [original modules](software-modules#original) because they have a larger set of packages compared to the new modules system. Similarly, switching to a package manager like Anaconda sometimes presents its own challenges. Instead, we will try the *direct installation* method and walk through the various obstacles to making this work.

## The problem: installing `rgdal`

The `R` programming language provides a very flexible package management system which compiles extra code maintained by the community. One popular package is the `rgdal` library, which provides bindings for the Geospatial Data Abstraction library. [CRAN describes this package here](https://cran.r-project.org/web/packages/rgdal/index.html).

The documentation on CRAN is fairly descriptive, and explains that `rgdal` depends on several other libraries *outside* of R, namely `PROJ` and `GDAL`. On a personal workstation, you might install this directly using the operating system package manager e.g. `apt-get install`. For security and stability, this is not an option on *Blue Crab*. To solve this problem, we have installed software in a central location which is available via [Lmod](https://lmod.readthedocs.io/en/latest/) and the `module` commands. 

Let's install `rgeos` and `rgdal` since these packages typically go together.

First we load the required modules, then we select an `R` module (note that you can use either the Intel or GCC compiler, which load different copies of R itself), and then we use R to install the package.

~~~
$ ml proj/5.1.0 gdal/2.4.0 
$ ml R/3.6.1
$ R
> install.packages('rgdal')
~~~

We see an unfortunate error message:

~~~
checking projects.h presence and usability... yes
checking PROJ.6: ... no
checking PROJ.4: epsg found and readable... no
Error: proj/epsg not found
Either install missing proj support files, for example
the proj-nad and proj-epsg RPMs on systems using RPMs,
or if installed but not autodetected, set PROJ_LIB to the
correct path, and if need be use the --with-proj-share=
configure argument.
ERROR: configuration failed for package 'rgdal'
~~~

## Solution: control the versions

When you see an error like this, the first step is read the documentation. The CRAN docs for [rgdal](https://cran.r-project.org/web/packages/rgdal/index.html) mention that version 1.4.1 attempts to use `PROJ` versions greater than 6. When we loaded the `proj` module above, the version was 5.1.0. While it is almost always best to use the latest version, the combinatorics spiral rapidly out of control when many users request different packages. As a result, the software tree might not always be current.

Let's try to use a slightly older version of `rgdal` that might be compatible with the existing `proj/5.1.0` module. To do this we should click the "Old sources: [rgdal archive](https://cran.r-project.org/src/contrib/Archive/rgdal/)" link on the [rgdal cran page](https://cran.r-project.org/web/packages/rgdal/index.html). Then, we grab the link for an older version and install it from source in the usual way.

~~~
$ ml proj/5.1.0 gdal/2.4.0 
$ ml R/3.6.1
$ R
> install.packages('https://cran.r-project.org/src/contrib/Archive/rgdal/rgdal_1.3-9.tar.gz',repos=NULL,type="source")
~~~

This time we get another error:

~~~
checking PROJ.4: epsg found and readable... no
Error: proj/epsg not found
Either install missing proj support files, for example
the proj-nad and proj-epsg RPMs on systems using RPMs,
or if installed but not autodetected, set PROJ_LIB to the
correct path, and if need be use the --with-proj-share=
configure argument.
ERROR: configuration failed for package 'rgdal'
~~~

In this case, the non-standard location of our `proj` installation is causing problems. We can overcome them by explicitly naming the share folders.

~~~
# continue from above
$ R
> install.packages('https://cran.r-project.org/src/contrib/Archive/rgdal/rgdal_1.3-9.tar.gz',repos=NULL,type="source",configure.args=c('--with-proj-include=/software/apps/proj/5.1.0/include','--with-proj-lib=/software/apps/proj/5.1.0/lib','--with-proj-share=/software/apps/proj/5.1.0/share'))
~~~

This works! Next we will install `rgeos` described [here](https://cran.r-project.org/web/packages/rgeos/index.html). Make sure to use the same version of R and the same compiler as before.

~~~
$ ml proj/5.1.0 gdal/2.4.0 geos/3.8.0
$ ml R/3.6.1
$ R
> install.packages('rgeos')
~~~

Of course we get another error but luckily we patience to spare.

~~~
icc -std=gnu11  -I"/software/apps/R/3.6.1/intel/18.0/lib64/R/include" -DNDEBUG -I/software/apps/geos/3.8.0/include -I"/home-net/home-4/rbradle8@jhu.edu/R/x86_64-pc-linux-gnu-library/3.6/intel/18.0/sp/include" -I/usr/local/include -I/software/centos7/usr/include   -fpic  -fPIC -qopenmp -O3 -ipo -multiple-processes=8  -c rgeos_wkt.c -o rgeos_wkt.o
rgeos_topology.c(85): error: identifier "GEOSMakeValid_r" is undefined
rgeos_topology.c(85): error: identifier "GEOSMakeValid_r" is undefined
      return( rgeos_topologyfunc(env, obj, id, byid, &GEOSMakeValid_r) ); 
                                                      ^
icc: error #10298: problem during post processing of parallel object compilation
compilation aborted for rgeos_topology.c (code 2)
make: *** [rgeos_topology.o] Error 2
make: *** Waiting for unfinished jobs....
ERROR: compilation failed for package 'rgeos'
~~~

This is no problem. First, we can goole part of the error: `error: identifier "GEOSMakeValid_r" is undefined`. The search is not very fruitful, but the first result shows us the geos changelog, which says that the variable `GEOSMakeValid_r` was added on `2019-02-25`. To sidestep this problem, we can search for an older version. If we return to the CRAN docs [here]() we can check for older versions by clicking the "Old sources: [rgeos archive](https://cran.r-project.org/src/contrib/Archive/rgeos/)". With some simple guesswork we can find an earlier version that works. We will install it from source in the usual way.

~~~
$ ml proj/5.1.0 gdal/2.4.0 geos/3.8.0
$ ml R/3.6.1
$ R
> install.packages('https://cran.r-project.org/src/contrib/Archive/rgeos/rgeos_0.5-2.tar.gz',repos=NULL,type="source")
~~~

## Recap

In this exercise we carefully controlled our package versions by consulting the CRAN docs and then compiling various packages directly from source. We had to do a little extra work to tell R where to find some software dependencies, but once we did, we were able to install our packages. It is important to note that *order matters* and this might not have worked if we installed `rgeos` first, because it might have produced an incompatible upstream dependency (`sp`). 

The process of controlling your versions has two knock-on benefits:

1. It allows you to overcome version-compatibility problems more easily.
2. It produces a more explicit record of the code you are using.

The second point above is critical for reproducibility. Even in the academic software world, versions are regularly updated and there are oftentimes mutual incompatibilities between specific sets of interdependent codes. Carefully managing these is essential to reproducing your work, and taking the time to stay as up-to-date as possible also ensures that you are using the most-correct software possible.
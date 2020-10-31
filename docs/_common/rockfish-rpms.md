---
layout: post
title: Rockfish local packages feature
---

# Installing local packages on RockFish

This feature, also known as "rootless installs" can be a handy trick for overcoming a `yum install` or `apt-get install` when moving your code to RockFish.

We are testing a tool which allows you to install local RPMs in your account on the RockFish cluster without root. In the following example, we will compile `proot` which requires a few libraries that are not part of the image or the software stack. We will install these libraries in `~/local` and access them using Lmod and an automatic private modulefile.

## Step 1. Collect requirements

This step cannot be automated. Assume that we know we need to install `libarchive-devel` and `talloc-devel`. We have collected the following RPM download URLs and written them to a file called `install.txt`.

~~~
cat > install.txt <<EOF
http://mirror.centos.org/centos/8/BaseOS/x86_64/os/Packages/libtalloc-2.2.0-7.el8.x86_64.rpm
http://mirror.centos.org/centos/8/BaseOS/x86_64/os/Packages/libtalloc-devel-2.2.0-7.el8.x86_64.rpm
http://mirror.centos.org/centos/8/BaseOS/x86_64/os/Packages/libarchive-3.3.2-8.el8_1.x86_64.rpm
http://mirror.centos.org/centos/8/PowerTools/x86_64/os/Packages/
EOF
~~~

*It is absolutely critical that these files are listed in the correct order.* They must also be able to find the correct versions of upstream dependencies in the system image. If you install in the wrong order, your packages will have broken links. If you use the wrong versions, you also get broken links. This feature does not take the place of a package manager. We are working on a more automatic solution for solving dependencies, however our intent is not to reinvent the wheel. The following feature is designed for a modest amount of packages, searchable on [`pkgs.org`](https://pkgs.org/), and installed without root.

## Step 2. Run the script

~~~
ml helpers
local-rpms.py -t install.txt -n proot
rm install.txt
~~~

Following installation, we use `ml own` to reveal our module, which has the same name as the "name" flag (`-n`) above. You can activeate your environment with `ml proot` and proceed to compile proot with the following method.

~~~
up=https://github.com/proot-me/proot
git clone $up
cd $(basename $up)
make -C src clean
make -C src loader.elf loader-m32.elf build.h
make -C src proot care
# pending error about uthash but proot works fine
~~~

The files created by the first step were placed in `~/local/<name>/usr` and we automatically generated a modulefile at `~/local/lmod/<name>.lua` with commom path manipulations (namely the one for pkgconf). You can extend or modify these to make sure your downstream code can access it.

Note that your libraries are likely to be dynamically linked for this directory, but you could feasibly use this method to compile static libraries and then later discard your lcoal environment.

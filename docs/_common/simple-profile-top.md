---
layout: post
title: Simple profiling with the "top" utility
---

*Blue Crab* has many different tools for performing massively parallel calculations. Oftentimes, you may need to monitor an example calculation to ensure that it is correctly using all of the right CPU resources you have requested from SLURM, or to watch the memory usage over time. The best way to do this is with `htop`. In this short guide we will explain the right procedure.

Note that we have removed `top` and `htop` from the login nodes because these are shared resources and most users should not need to monitor these nodes.

## Use interact

First, set up an interactive job with our `interact` command, which allows you to visit a compute node.

~~~
$ interact -p express -c 6 -t 60
Tasks:    1
Cores/task: 6
Total cores: 6
Walltime: 30:00
Reservation:   
Queue:    express
Command submitted: salloc -J interact -N 1-1 -n 1 -c 6 --time=30:00 -p express srun --pty bash
salloc: Granted job allocation 38634917
~~~

This command reserves hardware on our "express" partition which has a very low wait time (and a limit of 12 hours and 6 cores total).

## Start a screen on the compute node

Once your job is granted, your prompt will change to reflect your new location (`compute0123`). Create a [screen](https://www.howtoforge.com/linux_screen) with this command: `screen -S calculate`. Once you are inside the screen, start a calculation. This can be a manually-compiled code, or software from our modules system. Be sure to book the correct amount of cores. You may need to set the environment variable `OMP_NUM_THREADS` to use all six cores that we reserved.

While your code is running, use hold the `CTRL` key and press `A` then `D`. This detaches you from the screen. Your code will continue to run in the background.

You can see if you are either "attached" or "detached" to the screen by running `screen -ls`. To reattach to the screen, you should run `screen -r calculate`. If you follow [this guide](https://www.howtoforge.com/linux_screen) you can learn more about this tool. 

## Monitor your job while it runs

The `htop` program has a number of features which allow you to inspect the memory and CPU usage. The best-performing workflows will show a constant, steady usage of 100% on all reserved cores. Note that red bars or a red `D` tag on your processes indicate a high level of kernel overhead or a disk wait state. If you are interested in learning more about `htop`, please see [this guide](https://codeahoy.com/2017/01/20/hhtop-explained-visually/).

## Best practices

There are many ways to ensure that your code is using all of your requested resources efficiently, however the best method is to monitor your jobs for complete CPU usage and efficient memory usage. The guide above is a common method for checking your work before submitting a very large parallel calculation, and using `interact`, `screen`, and `htop` are easy ways to understand how this machine works.
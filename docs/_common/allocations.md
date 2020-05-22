---
layout: post
title: Allocation policy
---

This page outlines the current allocation policy for the *Blue Crab* cluster at [MARCC](https://www.marcc.jhu.edu/). You can inspect your current allocation with the `new_sbalance` command.

## Overview

The *Blue Crab* cluster controls access to the cluster with a ***shares*** system in which shares are disbursed on a quarterly basis (every three months, following the calendar year). One share is roughly equivalent to a single CPU-hour of computation on the machine. Shares are allocated for each research group. Your principal investigator (PI), for example, may have access to 100,000 shares, which are shared between members of their group. An allocation of 100,000 shares should allow your group to consume roughly 100,000 CPU-hours per quarter. *When you consume your allocation, you will no longer be able to use the hardware.* In the next section we will explain exactly how this works. Skip to the [balance](#balance) section to understand how to check your allocation balance.

## Recharge

In practice, the *Blue Crab* cluster experiences dynamic changes in demand which are largely the result of the ebb and flow of many small and large research projects simultaneously requesting hardware on the machine. Starting in Q2, 2020, we have implemented a *recharge* policy to help manage these competing demands for our computational resources.

An allocation is [a flow, not a stock](https://en.wikipedia.org/wiki/Stock_and_flow). Your group's allocation appears as a total number of hours available for computation, however these hours must be spent over a particular time period. Our accounting system is loosely calibrated to the quarterly system, specifically periods of three months that align with the calendar year. 

There are many disadvantages to resetting the allocations every quarter. First, users who quickly consume all of their hours would typically have to wait until the allocation resets to use the machine again. If the machine goes idle in the meantime, these resources are wasted. Second, groups which hoard their hours or fail to use them regularly throughout the quarter will typically try to use large amounts of their allocation shortly before the reset so that these hours do not disappear. Both of these usage patterns make it difficult to efficiently use the resource.

To more efficiently allocate the machine, your hours will slowly recharge after you use them. The larger your allocation, the faster they will recharge. Therefore, the allocation size (e.g. 100,000 hours per quarter) operates as a *set point* which controls the total rate of hours your group can consume. In the [balance](#sbalance) section below, we will show you how to check your allocation (the set point) along with your recent or apparent usage and your total usage for the quarter. 

If you consume 50,000 hours from an allocation with 100,000 hours, your "recent" or apparent usage will exponentially decay to zero, thereby returning your available hours to a full charge of 100,000 hours, with a half life of roughly one week (this may be subject to change). This system continuously resets your allocation, rather than resetting it all at once every three months.

## Recharge: practical implications

This policy ensures that groups who consume their hours quickly must only wait a short while before they can continue running jobs. It also ensures that groups which do not use their allocations do not cause the machine to go idle. The recharge creates a more gradual *use it or lose it* effect in which groups that regularly submit their jobs and consume their allocation will receive the greatest amount of recharged hours. While this policy prevents hoarding hours, it still ensures that every allocation is available to each group at any time, subject only to the current level of traffic on the cluster.

## Priority

In the guide above, we have explained that an allocation is a rate or flow. The total allocation size (e.g. 100,000 hours per quarter) acts as set point. The allocation sets the rate at which your hours recharge. Larger allocations recharge faster. We also use the allocation size to set your priority. Groups with large allocations are expected to consume these hours at a faster rate, hence they often need privileged access to the machine. *Blue Crab* determines priority, and hence wait times, by a score that is 50% related to how long you have been waiting (i.e. "first come, first serve") and 50% related to your allocation size.

Groups which consume a larger share of the *Blue Crab* hours, whether because they submit many jobs or regularly submit jobs, will experience lower priority and longer wait times when other groups who have used too few of their hours try to use the machine. This is a good thing! If your priority declines, it means you have responsibly consumed your hours, and it typically means you have also received a bonus for doing so, since recharge will often allow groups to consume their quarterly allocation in a shorter time. 

## Condos

If your PI has purchased hardware on *Blue Crab*, you have access to a separate "condo" account with higher priority and additional hours. When you run the `new_sbalance -u $USER` script, you will see more than one account. You can use the condo account a flag (`-A`) in your SLURM scripts to access this account.

## Checking the balance {#sbalance}

We have transitioned to a new account balance system. To preview the new system, run the `new_sbalance` command. Here is some example output. Some users may be associated with multiple projects.

~~~
+-----------------------------------------------+
|                                               |
|  Blue Crab Project: mcurie1                   |
|  ==========================                   |
|   1,250,000 shares (hours/Q)                  |
|     853,623 used (hours/Q)                    |
|     500,765 used recently (hours)             |
|         60% AVAILABLE (recent/shares)         |
|                                               |
|                  USER      USED    RECENT     |
|      adebier1@jhu.edu   635,685   358,788     |
|    * omoreno1@jhu.edu   217,938   141,978     |
|                                               |
|  see this link for more info:                 |
|  https://marcc-hpc.github.io/esc/allocations  |
|                                               |
+-----------------------------------------------+
~~~

This output tells us the following:

1. The total allocation size is 1.25 million hours per quarter.
2. This group has actually consumed over 800,000 hours so far.
3. Thanks to recharge, their apparent or recent usage is only about 500,000 hours. 
4. This means that they have 60% of their allocation, or roughly `1250000 - 500765 = 749235` hours *available for use now*. 

The "recent" usage is the apparent usage according to SLURM, and is the result of recharge. Your recent usage will decline if you do not use your hours, until the allocation is "fully charged" again. This display also shows the usage for each user. Remember that "recent" usage is the apparent usage" and sets the recharge rate. 

***High recent usage is rewarded with faster recharge rates while low recent usage gives you higher priority.***

Note that all members of the group are treated equally, and there is no mechanism for balancing priority between your group members. MARCC does not set policy between users. We expect group leaders to set this policy and encourage the members of the group to work together to use this machine.

## Wait times

Wait times must increase when you consume many hours or when you receive a very large "bonus" from submitting jobs regularly and other users need a chance to use the machine. You can check your wait time for recently completed or started jobs with:

~~~
waittime sacct -S 2020-04-01 -E 2020-04-30 -u user@school.edu
~~~

You can also see the wait times for currently running or pending jobs.

~~~
waittime squeue -p debug
~~~

Both of these commands simply wrap SLURM `sacct` and `squeue` commands so they can be customized further.

## Specialized Hardware

Since we do not presently charge a higher rate for specialized hardware, including nodes with extra memory or graphical processing units (GPUs), we ask that users ensure that their work *absolutely requires* these resources before scheduling jobs on them. This policy is subject to change. Users who repeatedly violate this policy will be banned from these resources.

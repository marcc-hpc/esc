---
layout: post
title: Job priority and the scheduler 
---

## Wait times on *Blue Crab* {#waits}

The *Blue Crab* cluster uses [SLURM](https://slurm.schedmd.com/documentation.html) to schedule your calculations. There are three factors which affect your wait time.

1. The cluster operates *almost entirely* according to a first-in-first-out (FIFO) queue, which means that jobs that request similar hardware will be scheduled in the order they are submitted. The job priority therefore increases linearly with the wait time (age) until it starts. 
2. The size, hardware, and partition will affect your wait time, particularly as demand can fluctuate significantly from week to week. A very large job will wait longer for resources than a small job. While SLURM reserves hardware for a large job, it attempts to schedule small jobs as "[backfill](https://slurm.schedmd.com/sched_config.html)" as long as they can finish before the large job is ready to start. Therefore, users can optimize their wait times by correctly estimating the amount of time they need. Keep in mind that if you underestimate the necessary time, however, your job will be terminated.
3. There is a minor exception to the FIFO rule for groups with very large allocations, who receive a small increase in their priority to ensure that they are able to consume their allocation each quarter when the cluster utilization is high. This feature is called [fairshare](https://slurm.schedmd.com/fair_tree.html).

*As a general rule, we do not guarantee low wait times for any group. Our cluster is optimized so that all users can consume their quarterly allocation. We encourage users to plan ahead to account for the wait time.*

### Reporting priority problems {#priority_problems}

In order for our staff to investigate priority questions, users who email `marcc-help@marcc.jhu.edu` must supply the following information:

1. The SLURM Job IDs for the jobs in question.
2. The path to the job script.
3. The current priority for the job according to `sprio -j N` where `N` is the job ID. This information disappears when the job ends. In order to understand wait times, we must have the priority numbers for your job.
4. If you believe that your priority values are inconsistent with other users in the queue, we require job IDs and priority (via `sprio -j N`) reports for those jobs while they are running. These jobs must have similar hardware requests for us to compare them properly. 

*We cannot investigate priority disputes without a direct apples-to-apples comparison between jobs along with their associated priority values.* We encourage users to keep in mind that the queue is not strictly based on wait time (see the [general rule above](#waits)) hence there may be minor variations in scheduling besides the natural variation in job size and demand.

In rare cases, your job may be stuck due to a misconfiguration ([see below](#stuck)), in which case our staff can help you request the correct hardware. 

### Jobs that are pending for too long {#stuck}

It is possible to submit a SLURM job that gets "stuck" in the pending state because it requires an impossible set of hardware according to the constraints on our system. While we have streamlined the system in recent months to prevent this, it most commonly occurs when users request more than 6 processors per GPU on our `gpuk80` partition. This violates the 6-processors-per-GPU rule that SLURM follows for that queue. The best solution is to ask for more GPUs or fewer CPUs per task. 

You can tell that your job is stuck when meets these conditions:

1. It is pending with the tag "Resources".
2. Other jobs with identical hardware requests and lower priority are scheduled first. 

If either of these conditions are not met, then your job is not stuck and it is simply waiting to get to the front of the line. The difference between a stuck job and a waiting job therefore depends strictly on its priority (given by `sprio`). Our staff can check pending jobs to confirm that the resource requests are accurate. After the next SLURM upgrade we will recover a feature that warns you when this is not the case. 
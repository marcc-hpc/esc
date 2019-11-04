---
layout: post
title: Parallel matlab
---

## Using a parallel pool

Matlab can be parallelized using [`parpool`](https://www.mathworks.com/help/parallel-computing/parpool.html;jsessionid=6f8bbc8006fc3130593220ed80cc). For many users, this can be as simple as replacing `for` loops with a `parfor` loop, as long as the problem is [pleasingly parallel](https://en.wikipedia.org/wiki/Embarrassingly_parallel) and the individual iterations of the loop do not depend on the outcome of the others.

### Best practices for `parfor` on *Blue Crab*

We recommend that users who require multiple, simultaneous parallel pools follow these steps when using `parfor` in order to avoid name collisions in the location where Matlab writes the temporary files required to organize its "workers" during the parallel calculation. This method can also be used if you would like these files to be written to another location for debugging or performance reasons.

1. Put the following command in your SLURM script. You can select a full path to the temporary location here. You are welcome to put these files in your home directory (`~/`) or the data directory (`~/data`) on our ZFS filesystem. If your calculation requires many parallel executions, we recommend against using the Lustre filesystem (`~/work`, `~/scratch`) for this kind of I/O because it will write many small files.

```
export TMPDIR=$(pwd)/matlab_tmp # or select another path
mkdir -p $TMPDIR
```

2. Instead of using the command `parpool(4)`, use the following commands to create one worker per processor. Note that your SLURM request should use `--cpus-per-task` to select the number of processors. Matlab does not use "tasks" which are [exclusively used for MPI jobs](https://slurm.schedmd.com/cpu_management.html). These commands must appear before your parallel `for` loop.

```
pc = parcluster('local');
nproc = str2num(getenv('SLURM_CPUS_PER_TASK'));
pc.NumWorkers = nproc;
job_folder = fullfile(getenv('TMPDIR'),getenv('SLURM_JOB_ID'));
mkdir(job_folder);
pc.JobStorageLocation = job_folder;
parpool(pc,nproc)
```

3. Use the `parfor` command anywhere in your script.

4. At the end of your script, remove the pool with these commands.

```
poolobj = gcp('nocreate');
delete(poolobj);
```

This procedure gives you the maximum control over the files required to support parallel execution.
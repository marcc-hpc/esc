---
layout: post
title: Blue Crab storage guidelines
---

These guidelines review the best practices for input/output (I/O) operations on the *Blue Crab* cluster at [MARCC](https://www.marcc.jhu.edu/getting-started/data-storage/).

## Concept: performance depends on bottlenecks

An old joke holds that high performance computing (HPC) is the [art of turning a CPU-bound program into an I/O-bound program](https://news.ycombinator.com/item?id=17206667). At MARCC we encourage all users to avoid this common trap by writing as much network- or memory- or compute-bound code as possible. 

The most common HPC clusters are generally optimized for very large *calculations* and not necessarily long-term storage of very large *data* sets. We encourage all MARCC users to optimize their workflow to minimize redundant or inefficient I/O profiles so that they can make the most of our facility.

## Storage interruptions on *Blue Crab* {#storage_problems}

1. **Lustre interruptions.** As of January, 2020, The *Blue Crab* cluster has resolved a number of usability issues that affected our Lustre filesystem, most of which were caused by IOPS-intensive workflows and a minor technical issue. We nevertheless recommend that all users continue to optimize their workflows according to the following guide.

2. <a id="data_unmount"></a>**Limited access to `~/data`.** As of October, 2020, we have removed access to the ZFS filesystem mounted at `~/data` from the compute nodes because many users were continuing to use this system for direct computation and thereby harming its integrity. All direct calculations should be performed on the Lustre filesystem at `~/work` (shared within a group) and `~/scratch` (individual), both of which are not backed up. No executables should be stored in `~/data` or on Lustre. See guidelines below for storing executables in `~/code` or a shared space in `~/work/code`. *If you need to access data from the `~/data` filesystem from a compute job, you must first copy the data to scratch from the login nodes or a data transfer node.*

## Optimizing storage

*If you are experiencing a sporadic I/O error, please read the following guide for diagnosing and correcting the problem.* Note that a persistent I/O error (one that occurs every single time you run your program) is more likely to be due to user error, in which case you should try debugging your code first before seeking assistance from our support team.

### Step 0: Manage small-file I/O on Lustre

If you *already know* that you are trying to write **many** (>1000) **small files** (<500MB) files on **Lustre** (`~/work` and `~/scratch`) then you can use this one easy trick to improve performance. Find the directory that will hold these files, and then run the following command.

~~~
lfs setstripe -c 1 ./path/to/small/file/writes
~~~

This will signal to our high-performance Lustre system that you will *not* be performing large block I/O and it will take steps to ease the burden of these small files on our system. The command above will set the concurrency to one. Lustre typically tries to take large files and fragment them across many servers to improve performance and reliability. This is counterproductive when users write many small files, hence the recommendation to write them to a single server.

#### Where to compile code {#compile_where}

Additionally, you **should not compile or store executables** on Lustre (`~/scratch` and `~/work`). These should be stored in your home directory or data directory (`~/` or specifically `~/code`) which provides a read-optimized ZFS filesystem that is appropriate for codes. 

In October, 2020, we have also provided a `~/work/code` directory for **shared codes**. This space is a read-optimized filesystem shared between members of a single group. This change compensates for the removal of the shared `~/data` mount from all compute nodes (see above, [data storage guidelines](#data_unmount)). 

Any codes or scripts which are shared between members of a group should be placed in `~/work/code` while individual codes should be located in `~/` or `~/code`.

### Step 1: **Determine your I/O profile** {#profile}

The first step required to overcome a storage-related difficulty is to characterize your "I/O profile" by estimating the following behaviors of your program:

- How often do I read or write files? Are multiple processors performing I/O at the same time?
- Which filesystems (i.e., specific paths) are hosting the data that I am reading or writing?
- How large are these files?

In short, you must first estimate the *frequency*, *size*, and *location* for both *reads* and *writes* for the duration of your program. The specific details may make a dramatic difference.

### Step 2: **Optimize your I/O** {#optimize}

While profiling is useful in general, this step is specific to *Blue Crab* and will help determine the best solution to your storage error. Before continuing, let's review the two filesystems.

- The *Lustre* filesystem is a high-performance scratch system mounted on `~/work` and `~/scratch`.
- The *ZFS* filesystem is a standard filesystem which hosts `~/` (except for the Lustre folders above) along with `~/data` and possibly `~/work-zfs` for some groups.

Once you have collected the size, frequency, and paths for reads and writes, consult the following list of "cases" for more specific advice. Note that "small" and "big" refer to the combination of frequency and file size. For example, if I have to write 100 separate 1 MB files every 6 hours from 5 simultaneous jobs, then your writes would total 3GB per day over 2000 files. 

In the following categories, we say that any workflow that creates less than 10GB per day in persistent storage from all workers is "small" while everything else is "big" for the purposes of our estimates. If your program only "keeps" 10GB per day but needs to read and write 100GB then it has "big" I/O needs. Similarly, if your program writes 500,000 small files, we also say that it has "big" I/O because many small file transactions can often be more costly than writing a single large file.

We refer to the size of the I/O by *bandwidth* and not the final size on disk. If you are writing 100GB in a single day, that counts as large I/O while accumulating 100GB of data steadily over 6 months is considered small.

#### Case 1. Small reads and writes on Lustre

If you experience an error writing or reading a very modest amount of data on Lustre, we recommend further investigation. Lustre typically only fails on small writes if you have a severly high number of file `stat` operations. If this is the case, your workflow would be better served by moving these operations to `/dev/shm` (a shared memory location) or by reformulating your workflow. It's possible that a collective slowdown of Lustre caused by high IOPS from other users is also causing problems with modest I/O on Lustre, in which case we recommend running at least a few different tests of your code to see if the problem is sporadic. If the problem is sporadic, it is more likely to be traffic. 

#### Case 2. Large writes on Lustre

Nearly all of the I/O from compute nodes on *Blue Crab* should occur on the Lustre filesystem which is specifically optimized to write and read large files. If you encounter errors with large write operations on Lustre, it may be due to a fundamental limitation. Since we have recently [removed the `~/data` mount](#data_unmount) from the compute nodes due to technical limits, the only alternatives are to write to your home directory, which has a limited size, or to an optional `~/work-zfs` purchased by some groups. If you experience an I/O bottleneck when writing large files on Lustre, we recommend reformulating your workflow rather than moving to the home directory or `~/work-zfs`. You can reduce the frequency  that you write the data, or alternately, you can consult external documentation (e.g. from [NICS](https://www.nics.tennessee.edu/computing-resources/file-systems/lustre-striping-guide)) to improve your Lustre striping strategy. If you are writing many separate files but the total size is not very large, see case #3 below.

#### Case 3. Writing many files on Lustre

Lustre is optimized for large block I/O operations and has a performance penalty when writing many tiny files. We strongly recommend reducing the number of separate files by altering your code. If you absolutely cannot do this, it may be useful to move some of these operations to memory to reduce the total output. This can be done by modifying your code dorectly, or by using `/dev/shm`. 

#### Case 3. Large reads on ZFS

If you have a large number of files or a large amount of input data for your work, and this data is useful for more than one separate calculation, we recommend keeping it on the ZFS filesystem. This filesystem is designed for long-term storage of your data whereas Lustre is designed for temporary storage, hence the name. If you are experiencing errors while reading a large data set on ZFS, it is best to optimize your use of memory to reduce the load rather than moving the data to Lustre.

#### Case 4. Large writes on ZFS

The ZFS filesystem is not optimized for a large amount of writing, especially if you are only writing temporary data. Try migrating the I/O intensive portions of your work to Lustre. It is also best to reduce the number of files you write, along with the frequency of times you write to the file. Lustre performs best with large block operations, so we encourage you to cache your data in memory and then write it all at once in large portions.

## General guidance

There are a few special guidelines for working on our system. These recommendations are based on [best practices helpfully summarized here](https://hpcf.umbc.edu/general-productivity/lustre-best-practices/) and [here](https://www.nas.nasa.gov/hecc/support/kb/lustre-best-practices_226.html).

#### Do not compile code on Lustre and avoid repetitive open/close operations

Compiling code requires many small file operations and asks many questions about each file. Lustre optimizes large data I/O by separating the act of writing the data from the act of describing it with so-called *metadata*. Lustre will not perform well when your program repeatedly asks for metadata by checking the existence of a file, listing the contents of a directory (especially with `ls -l`), or performing many repetitive 

We recommend that you store source code and compile your code on our read-optimized ZFS file systems (at `~/` and `~/work/code`, see [above](#compile_where)). It is also sub-optimal to run your compiled code from Lustre (see below).

#### Avoid using Lustre for executables

In addition to avoiding code compilation (see above), we also recommend storing your executables on ZFS instead of Lustre. Executables, which are typically much smaller than the files that Lustre was optimized for, will run slower on Lustre. According to [this documentation](https://www.nas.nasa.gov/hecc/support/kb/lustre-best-practices_226.html) you should also avoid copying new executables over existing ones.

#### Parallel Pools in Matlab

If you wish to use multiple parallel pools in your Matlab code, please see [these instructions](matlab-parpool).

### Closing thoughts

We encourage all users who are having difficulty with I/O to read the guidelines above to help characterize their workflow. Our staff can be reached at [`marcc-help@marcc.jhu.edu`](mailto:marcc-help@marcc.jhu.edu) for further assistance.


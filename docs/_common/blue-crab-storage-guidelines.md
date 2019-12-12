---
layout: post
title: Blue Crab storage guidelines
---

These guidelines review the best practices for input/output (I/O) operations on the *Blue Crab* cluster at [MARCC](https://www.marcc.jhu.edu/getting-started/data-storage/).

## Concept: performance depends on bottlenecks

An old joke holds that high performance computing (HPC) is the [art of turning a CPU-bound program into an I/O-bound program](https://news.ycombinator.com/item?id=17206667). At MARCC we encourage all users to avoid this common trap by writing as much network- or memory- or compute-bound code as possible. 

The most common HPC clusters are generally optimized for very large *calculations* and not necessarily long-term storage of very large *data* sets. We encourage all MARCC users to optimize their workflow to minimize redundant or inefficient I/O profiles so that they can make the most of our facility.

## Storage interruptions on *Blue Crab* {#storage_problems}

As of Fall 2019 the *Blue Crab* cluster is experiencing a very high level of I/O traffic as well as a technical issue with our Lustre (scratch) filesystem which requires an extensive kernel and driver upgrade (estimated for December 2019). 

*If you are experiencing a sporadic I/O error, please read the following guide for diagnosing and correcting the problem.* Note that a persistent I/O error (one that occurs every single time you run your program) is more likely to be due to user error, in which case you should try debugging your code first.

### Step 0: Write small files

If you *already know* that you are trying to write **many (>1000) small (<500MB) files on Lustre (`~/work` and `~/scratch`)** then you can use this one easy trick to improve performance. Find the directory that will hold these files, and then run the following command.

~~~
lfs setstripe -c 1 ./path/to/small/file/writes
~~~

This will signal to our high-performance Lustre system that you will *not* be performing large block I/O and it will take steps to ease the burden of these small files on our system. The command above will set the concurrency to one. Lustre typically tries to take large files and fragment them across many servers to improve performance and reliability. This is counterproductive when users write many small files, hence the recommendation to write them to a single server.

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

If you experience an error writing or reading a very modest amount of data on Lustre, it is safe to try to move your workflow to the ZFS system. We recommend the `~/data` location because the home directory has a 20GB quota. 

Since your input files (those that your program reads) are small, it should be easy to move your workflow. If your problem persists, it might be a problem with your application, and not the filesystem. *Our ZFS system has much lower performance than our Lustre system, therefore we do not recommend using this system for all of your high-bandwidth file operations. We only recommend using it as a temporary, stopgap measure.* 

#### Case 2. Large writes on Lustre

If you experience an error writing a large amount of data to Lustre, we do not encourage you to move your workflow to ZFS (case #1 above) because this may negatively impact that filesystem. Instead, the best option is to determine whether you can either reduce the frequency that you write your data. If you are writing many separate files but the total size is not very large, see case #3 below. If you cannot reduce your total output size, it may be useful to move some of your write operations to our ZFS filesystem.

#### Case 3. Writing many files on Lustre

Lustre is optimized for large block I/O operations and has a performance penalty when writing many tiny files. We strongly recommend reducing the number of separate files by altering your code. If you absolutely cannot do this, it may be useful to move some of these operations to memory to reduce the total output. As a last resort, you may try moving some of your write operations to ZFS bearing in mind that if many people do this, performance will decline for all users.

#### Case 3. Large reads on ZFS

If you have a large number of files or a large amount of input data for your work, and this data is useful for more than one separate calculation, we recommend keeping it on the ZFS filesystem. This filesystem is designed for long-term storage of your data whereas Lustre is designed for temporary storage, hence the name. If you are experiencing errors while reading a large data set on ZFS, it is best to optimize your use of memory to reduce the load rather than moving the data to Lustre.

#### Case 4. Large writes on ZFS

The ZFS filesystem is not optimized for a large amount of writing, especially if you are only writing temporary data. Try migrating the I/O intensive portions of your work to Lustre. It is also best to reduce the number of files you write, along with the frequency of times you write to the file. Lustre performs best with large block operations, so we encourage you to cache your data in memory and then write it all at once in large portions.

## General guidance

There are a few special guidelines for working on our system. These recommendations are based on [best practices helpfully summarized here](https://hpcf.umbc.edu/general-productivity/lustre-best-practices/) and [here](https://www.nas.nasa.gov/hecc/support/kb/lustre-best-practices_226.html).

#### Do not compile code on Lustre and avoid repetitive open/close operations

Compiling code requires many small file operations and asks many questions about each file. Lustre optimizes large data I/O by separating the act of writing the data from the act of describing it with so-called *metadata*. Lustre will not perform well when your program repeatedly asks for metadata by checking the existence of a file, listing the contents of a directory (especially with `ls -l`), or performing many repetitive 

We recommend that you store source code and compile your code on our ZFS system (at `~/`, `~/data`, and `~/work-zfs`). It is also sub-optimal to run your compiled code from Lustre (see below).

#### Avoid using Lustre for executables

In addition to avoiding code compilation (see above), we also recommend storing your executables on ZFS instead of Lustre. Executables, which are typically much smaller than the files that Lustre was optimized for, will run slower on Lustre. According to [this documentation](https://www.nas.nasa.gov/hecc/support/kb/lustre-best-practices_226.html) you should also avoid copying new executables over existing ones.

#### Parallel Pools in Matlab

If you wish to use multiple parallel pools in your Matlab code, please see [these instructions](matlab-parpool).

### Closing thoughts

We encourage all users who are having difficulty with I/O to read the guidelines above to help characterize their workflow. Our staff can be reached at [`marcc-help@marcc.jhu.edu`](mailto:marcc-help@marcc.jhu.edu) for further assistance.


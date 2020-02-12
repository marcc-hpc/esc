---
layout: post
title: "Tensorflow: latest"
---

While we prepare for an upgrade to CUDA 10.0, uses who require tensorflow are welcome to install the CPU-only version using `pip` according to the [installation instructions](). Tensorflow often requires the penultimate version of Python, therefore we recommend that you install a custom conda environment according to our guide here](software-environments#conda). When you follow this guide, use the following requirements.

~~~
dependencies:
  - python==3.7
  - pip
  - pip:
    - tensorflow
~~~

We strongly recommend benchmarking your codes that use a non-GPU tensorflow to ensure that you are wisely spending your allocation. A full upgrade to CUDA 10 along with instructions for using tensorflow will be released shortly.

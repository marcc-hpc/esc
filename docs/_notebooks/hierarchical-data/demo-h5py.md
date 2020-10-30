---
layout: post
shortname: "demo-h5py"
title: ""
tags:
    - python
    - notebook
--- 
# Objectives

In this notebook we will demonstrate the use of `h5py` for writing complex data
structures. This tutorial follows the [h5py
documentation](https://docs.h5py.org/en/stable/quick.html) closely. 
 
# Requirements

We will build a python environment in the usual way. You are welcome to use a
virtual environment instead. 

**In [1]:**

{% highlight python %}
# start with Anaconda or a virtualenv
# on MARCC: ml anaconda
# then install with: conda update --file reqs.yaml -p ./path/to/new/env
# then use the environment with: conda activate -p ./path/to/new/env
# then start a notebook with: jupyter notebook
# the reqs.yaml file contents are:
_ = """
name: demo_h5py
dependencies:
  - python==3.8.3
  - conda-forge::ipdb
  - h5py
  - notebook
  - pip
  - pip:
    - scipy
    - numpy
    - pyyaml
    - ruamel
    - mpi4py
"""
{% endhighlight %}
 
# Use cases

## Case 1: Writing and reading arrays

The documentation says that h5py files can hold two kinds of information. Groups
work like dictionaries, and datasets work like NumPy arrays. 

**In [37]:**

{% highlight python %}
import numpy as np
import h5py
{% endhighlight %}

**In [38]:**

{% highlight python %}
eg_dims = (5,4,3)
data = np.random.rand(*eg_dims)
{% endhighlight %}

**In [39]:**

{% highlight python %}
# check the docs on the create_dataset function before we start
h5py.File.create_dataset?
# we see that shape is required if the data is not provided
{% endhighlight %}

**In [24]:**

{% highlight python %}
# make a new file (we always use a context manager)
# using a context manager allows us to skip the fp.close at the end without issue
with h5py.File('my_data.hdf5','w') as fp:
    # create a dataset. recall that the shape and dtype are detected
    fp.create_dataset('result_1',data=data)
{% endhighlight %}

**In [41]:**

{% highlight python %}
# read the file. it prefers a read mode
with h5py.File('my_data.hdf5','r') as fp:
    print(fp)
    # the object acts like a dict
    print(list(fp.keys()))
    # we can check the dimensions of the result object
    result = fp['result_1']
    print(result.shape)
    # we can validate the dimensions
    if not fp['result_1'].shape==eg_dims: 
        raise ValueError('wrong dimensions!')
    # we can slice the arrays per usual
    subset = result[:2,...,:2]
    print(subset.shape)
{% endhighlight %}

    <HDF5 file "my_data.hdf5" (mode r)>
    ['result_1']
    (5, 4, 3)
    (2, 4, 2)

 
## Case 2: Using the hierarchy

In short, a hierarchical data format contains a POSIX-like filesystem. 

**In [56]:**

{% highlight python %}
# create a new file
with h5py.File('richer_data.hdf5','w') as fp:
    # create a group
    grp = fp.create_group('timeseries')
    for i in range(10):
        # generate some fake timseries data
        ts = np.random.rand(1000,2)
        grp.create_dataset(str(i),data=ts)
{% endhighlight %}

**In [58]:**

{% highlight python %}
# read the data
with h5py.File('richer_data.hdf5','r') as fp:
    # create a group
    print(fp['timeseries/0'])
    print(np.array(fp['timeseries/0']))
{% endhighlight %}

    <HDF5 dataset "0": shape (1000, 2), type "<f8">
    [[0.60755507 0.25658715]
     [0.41480807 0.29707645]
     [0.4591176  0.2710604 ]
     ...
     [0.01291844 0.95650034]
     [0.95477923 0.65849145]
     [0.94202922 0.08009092]]

 
## Case 3: Using metadata (attributes) 

**In [73]:**

{% highlight python %}
# simulate some metadata
meta = {
    'model':{
        'k0':1.2,'k1':1.3,'tau':0.01,'mass':123.},
    'integrator':{
        'dt':0.001,'method':'rk4',},
    'data':{
        'source_path':'path/to/data',
        'host':'bluecrab',},}
{% endhighlight %}

**In [74]:**

{% highlight python %}
# or use yaml (not part of the standard library)
import yaml
text = yaml.dump(meta)
meta = yaml.load(text,Loader=yaml.SafeLoader)
print(text)
{% endhighlight %}

    data:
      host: bluecrab
      source_path: path/to/data
    integrator:
      dt: 0.001
      method: rk4
    model:
      k0: 1.2
      k1: 1.3
      mass: 123.0
      tau: 0.01
    


**In [77]:**

{% highlight python %}
# serialize the data
import json
meta_s = json.dumps(meta)
print(meta_s)

{% endhighlight %}

    {"data": {"host": "bluecrab", "source_path": "path/to/data"}, "integrator": {"dt": 0.001, "method": "rk4"}, "model": {"k0": 1.2, "k1": 1.3, "mass": 123.0, "tau": 0.01}}


**In [79]:**

{% highlight python %}
# create a new file, this time with metadata
with h5py.File('richer_data.hdf5','w') as fp:
    # add the metadata
    fp.create_dataset('meta',data=np.string_(meta_s))
    # create a group
    grp = fp.create_group('timeseries')
    for i in range(10):
        # generate some fake timseries data
        ts = np.random.rand(1000,2)
        grp.create_dataset(str(i),data=ts)
{% endhighlight %}

**In [110]:**

{% highlight python %}
# extract the data
with h5py.File('richer_data.hdf5','r') as fp:
    #! first we get an h5py object
    #! meta_s = fp['meta']
    #! next we realize our docs are old
    #! meta_s = fp['meta'].value
    #! next we get a byte
    #! meta_s = fp['meta'][()]
    meta_s = fp['meta'][()].decode()
    #! print(meta_s)
    # unpack the data
    meta = json.loads(meta_s)
    print(yaml.dump(meta))
    # get an item in the timeseries
    result = fp['timeseries/8']
    # cast this as an array
    print(np.array(result))
{% endhighlight %}

    data:
      host: bluecrab
      source_path: path/to/data
    integrator:
      dt: 0.001
      method: rk4
    model:
      k0: 1.2
      k1: 1.3
      mass: 123.0
      tau: 0.01
    
    [[0.23862044 0.31504744]
     [0.67640219 0.47622231]
     [0.20556927 0.1666343 ]
     ...
     [0.48551332 0.84930619]
     [0.77959628 0.74633257]
     [0.31485861 0.30996229]]

 
## Case 4: Hashing your data

Or, how do I organize the data? The following is a poor-man's database. 

**In [114]:**

{% highlight python %}
import hashlib
# serialize the data with some special flags
# read about stability: https://stackoverflow.com/a/22003440
meta_s = json.dumps(meta,
    ensure_ascii=True,sort_keys=True,default=str)
hashcode = hashlib.sha1(meta_s.encode()).hexdigest()[:10]
# we can save unique files with guaranteed-unique names
print('data_%s.h5py'%hashcode)
{% endhighlight %}

    data_eba300792d.h5py


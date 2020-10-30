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

We will build a python environment in the usual way. Our goal is to install the
`h5py` package. 

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

**In [2]:**

{% highlight python %}
# alternative virtual environment instructions
_ = """
# get a python if you are on a cluster
$ ml python/3.8.6
# install a virtual environment
$ python -m venv ./env
# activate the environment
$ source ./env/bin/activate
# install packages as needed
(env) $ pip install h5py
"""
{% endhighlight %}
 
# Use cases

## Case 1: Writing and reading arrays

The documentation says that h5py files can hold two kinds of information. Groups
work like dictionaries, and datasets work like NumPy arrays. 

**In [3]:**

{% highlight python %}
import numpy as np
import h5py
{% endhighlight %}

**In [4]:**

{% highlight python %}
eg_dims = (5,4,3)
data = np.random.rand(*eg_dims)
{% endhighlight %}

**In [5]:**

{% highlight python %}
# check the docs on the create_dataset function before we start
h5py.File.create_dataset?
# we see that shape is required if the data is not provided
{% endhighlight %}

**In [6]:**

{% highlight python %}
# make a new file (we always use a context manager)
# using a context manager allows us to skip the fp.close at the end without issue
with h5py.File('my_data.hdf5','w') as fp:
    # create a dataset. recall that the shape and dtype are detected
    fp.create_dataset('result_1',data=data)
{% endhighlight %}

**In [7]:**

{% highlight python %}
# check to see that the file exists using cell magic in Jupyter
! ls
{% endhighlight %}

    demo-h5py-v2.ipynb my_data.hdf5       test
    demo-h5py.ipynb    richer_data.hdf5


**In [8]:**

{% highlight python %}
# read the file. it prefers a read mode
with h5py.File('my_data.hdf5','r') as fp:
    # pring the object (it tells us that it is an HDF5 file)
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

**In [9]:**

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

**In [10]:**

{% highlight python %}
# read the data
with h5py.File('richer_data.hdf5','r') as fp:
    # create a group
    print(fp['timeseries/2'])
    # you can print the data by 
    # casting it: print(np.array(fp['timeseries/2']))
{% endhighlight %}

    <HDF5 dataset "2": shape (1000, 2), type "<f8">

 
## Question: how can we add columns to arrays?

The best way to include metadata in your file is to attach it directly to an
Array. You can do this by using the numpy
[dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html) 

**In [11]:**

{% highlight python %}
# first we found a nice example from the docs
np.array?
{% endhighlight %}

**In [12]:**

{% highlight python %}
# in this example we set the string and 
# float types, along with names, for our colums
data_with_cols = np.array(
    [('ryan',2.5),('jane',4.0)],
    dtype=[('student','<S4'),('grades','<f4')])
{% endhighlight %}

**In [13]:**

{% highlight python %}
# the strings are converted to bytes
print(data_with_cols)
# we can review the column names here
print(data_with_cols.dtype.names)
# and this result can be stored directly with h5py
# in the next section we add unstructured data
# (possibly metadata) to the h5py file
{% endhighlight %}

    [(b'ryan', 2.5) (b'jane', 4. )]
    ('student', 'grades')


**In [14]:**

{% highlight python %}
# we can now add this to our file
# the following demonstrates the append feature
# and we also use try/except to rewrite a dataset if it exists
with h5py.File('richer_data.hdf5','a') as fp:
    if 'student_data' in fp: del fp['student_data']
    fp.create_dataset('student_data',data=data_with_cols)
{% endhighlight %}

**In [15]:**

{% highlight python %}
# check the file
with h5py.File('richer_data.hdf5','r') as fp:
    print(list(fp.keys()))
{% endhighlight %}

    ['student_data', 'timeseries']


**In [16]:**

{% highlight python %}
# here is an alternate method for formulating the array
# start with columns, ordered by rows, with distinct types
students = np.array(['ryan','jane','nikhil']).astype('<S16')
grades = np.array([2.5,4.0,3.6])
students,grades
{% endhighlight %}




    (array([b'ryan', b'jane', b'nikhil'], dtype='|S16'), array([2.5, 4. , 3.6]))



**In [17]:**

{% highlight python %}
# if you transpose this without a dtype, h5py will not tolerate the U6 (unicode) type
np.array(np.transpose((students,grades)))
{% endhighlight %}




    array([[b'ryan', b'2.5'],
           [b'jane', b'4.0'],
           [b'nikhil', b'3.6']], dtype='|S32')



**In [18]:**

{% highlight python %}
n_rows = students.shape[0]
dt = np.dtype([('student','<S32'),('grade','<f4')])
student_data = np.empty(n_rows,dtype=dt)
student_data['student'] = students
student_data['grade'] = grades
{% endhighlight %}

**In [19]:**

{% highlight python %}
# our data is now structured by dtype
student_data
{% endhighlight %}




    array([(b'ryan', 2.5), (b'jane', 4. ), (b'nikhil', 3.6)],
          dtype=[('student', 'S32'), ('grade', '<f4')])



**In [20]:**

{% highlight python %}
# we can select one column or row
print(student_data['grade'])
print(student_data[1])
{% endhighlight %}

    [2.5 4.  3.6]
    (b'jane', 4.)

 
Users interested in more data science-oriented structures are encouraged to
check out [pandas](https://pandas.pydata.org/). 
 
## Case 3: Using metadata (attributes) 

**In [21]:**

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

**In [22]:**

{% highlight python %}
# or use yaml (not part of the standard library) (pip install pyyaml)
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
    


**In [23]:**

{% highlight python %}
# serialize the data
import json
meta_s = json.dumps(meta)
print(meta_s)
{% endhighlight %}

    {"data": {"host": "bluecrab", "source_path": "path/to/data"}, "integrator": {"dt": 0.001, "method": "rk4"}, "model": {"k0": 1.2, "k1": 1.3, "mass": 123.0, "tau": 0.01}}


**In [24]:**

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

**In [25]:**

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
    # you can view the data by
    # casting it as an array: print(np.array(result))
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
    


**In [26]:**

{% highlight python %}
with h5py.File('richer_data.hdf5','r') as fp:
    this = fp['meta'][()].decode()
    meta = json.loads(this)
{% endhighlight %}

**In [27]:**

{% highlight python %}
import pprint
pprint.pprint(meta,width=10)
{% endhighlight %}

    {'data': {'host': 'bluecrab',
              'source_path': 'path/to/data'},
     'integrator': {'dt': 0.001,
                    'method': 'rk4'},
     'model': {'k0': 1.2,
               'k1': 1.3,
               'mass': 123.0,
               'tau': 0.01}}

 
## Case 4: Hashing your data

Or, how do I organize the data? The following is a poor-man's database. 

**In [28]:**

{% highlight python %}
# if you change the metadata, you change the hash
meta['data']['host'] = 'rockfish'
{% endhighlight %}

**In [29]:**

{% highlight python %}
import hashlib,json
# serialize the data with some special flags
# read about stability: https://stackoverflow.com/a/22003440
meta_s = json.dumps(meta,
    ensure_ascii=True,sort_keys=True,default=str)
hashcode = hashlib.sha1(meta_s.encode()).hexdigest()[:10]
# we can save unique files with guaranteed-unique names
print('data_%s.h5py'%hashcode)
{% endhighlight %}

    data_8c9c7dcc57.h5py

 
To conclude: you can treat each file as a metaphorical row in a database. The
filesystem thereby acts as a large parallel database. This saves the effort of
he traditional wrappers, handlers, validation, etc, that a database requires.
You only have to perform some modest file management. In the future we can
discuss more fully-featured database solutions. 

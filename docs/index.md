---
layout: home
title: ""
---

The "elements of scientific computing" (`ESC`) is a collection of workshop examples and answers to common scientific computing questions. This resource should be generally useful to academic researchers who are using scientific computing methods or high-performance computing (HPC) in their work. These notes were compiled by [Ryan Bradley](http://scattershot.info) for users of the Maryland Advanced Research Computing Center ([MARCC](https://www.marcc.jhu.edu)) and we therefore refer to our largest machine, *Blue Crab*, throughout the text.

## Documentation sites

Our documentation is currently in flux. The following guide should help you find the right information.

1. Questions about MARCC resources can be found on the main site:<br>[`https://www.marcc.jhu.edu`](https://www.marcc.jhu.edu)
2. This "short course" is a good introduction to the machine for new users:<br>[`https://marcc-hpc.github.io/tutorials/`](https://marcc-hpc.github.io/tutorials/)
3. The current site includes a set of [common tasks](#common-tasks) and materials for the [tutorial series](#tutorial-series) below.

{% if site.data.tutorials %}

## Tutorial Series

The following notes cover some of our work in the [ongoing tutorial series](https://www.marcc.jhu.edu/training/tutorial-series/).

<ol>
{% for tutorial in site.data.tutorials %}
<li><a href="{{ site.baseurl }}{{ tutorial.link }}">
	<strong>{{ tutorial.name | capitalize }}</strong></a>: {{ tutorial.info }}</li>
{% endfor %}
</ol>
{% endif %}

## Common tasks
<ul>
{% for task in site.data.common %}
<li><a href="{{task.link}}">{{task.name}}</a></li>
{% endfor %}
</ul>


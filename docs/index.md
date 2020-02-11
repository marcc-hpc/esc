---
layout: home
title: ""
---

The "elements of scientific computing" (`ESC`) is a collection of workshop examples and answers to common scientific computing questions. This resource should be generally useful to academic researchers who are using scientific computing methods or high-performance computing (HPC) in their work. These notes were compiled by [Ryan Bradley](http://scattershot.info) for users of the Maryland Advanced Research Computing Center ([MARCC](https://www.marcc.jhu.edu)) and we therefore refer to our largest machine named *Blue Crab* throughout the text. Material for the [tutorial series] is listed below.

## Documentation sites

Our documentation is currently in flux. The following guide should help you find the right information.

1. Questions about MARCC resources can be found on the main site:<br>[`https://www.marcc.jhu.edu`](https://www.marcc.jhu.edu)
2. This "short course" is a good introduction to the machine for new users:<br>[`https://marcc-hpc.github.io/tutorials/`](https://marcc-hpc.github.io/tutorials/)
3. The current site includes a set of [core documentation](#docs), [common workflows](#common-tasks), and materials for the [tutorial series](#tutorial-series) below.

## Core Documentation for *Blue Crab* {#docs}

The following guides cover the most critical features of the *Blue Crab* cluster. Most users will find the right software in our *software modules* system, while users who require Python, R, virtual environments, or Anaconda should consult the guide on *software environments* linked below. 

<ol>
{% for task in site.data.common.core %}
<li><a href="{{task.link}}">{{task.name}}</a></li>
{% endfor %}
</ol>

If you cannot find what you need in the guides on this page, please contact our support staff at <code><nobr><a href="marcc-help@marcc.jhu.edu">marcc-help@marcc.jhu.edu</a></nobr></code> for additional guidance.

## Common tasks and workflows {#common-tasks}

The following list provides answers to frequently asked questions and common workflows. These answers are specific to *Blue Crab* and designed to help users migrate their local development and calculations to our high-performance computing (HPC) environment.

<ul>
{% for task in site.data.common.common %}
<li><a href="{{task.link}}">{{task.name}}</a></li>
{% endfor %}
</ul>

*Note that this site is currently under construction as we migrate some of our documentation here.*

{% if site.data.tutorials %}

## Tutorial Series {#tutorial-series}

The following notes cover some of our work in the [ongoing tutorial series](https://www.marcc.jhu.edu/training/tutorial-series/).

<ol>
{% for tutorial in site.data.tutorials %}
<li><a href="{{ site.baseurl }}{{ tutorial.link }}">
	<strong>{{ tutorial.name | capitalize }}</strong></a>: {{ tutorial.info }}</li>
{% endfor %}
</ol>

{% endif %}

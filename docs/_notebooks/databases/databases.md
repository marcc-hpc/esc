---
layout: post
shortname: "databases"
title: ""
tags:
    - python
    - notebook
--- 
# Using Databases

This exercise will introduce relational and unstructured databases. We will use
a dataset kindly shared by Jan Aerts, the leader of the Data Visualization Lab
at KU Leuven, Belgium. He introduces Mongo in [this exercise](http://vda-
lab.github.io/2016/04/mongodb-exercises) however we will start with relational
databases.

To complete this tutorial, we will use a few different small datasets available
alongside this notebook. We will use the following environment.

```
# environment for elements of scientific computing (esc) database tutorial
name: esc03
dependencies:
  - python=3.7
  - conda-forge::mongo-tools
  - conda-forge::sqlite
  - pymongo
  - sqlalchemy
  - nb_conda_kernels
  - jupyter
  - jupyterlab
  - pip
```

You can build this environment from [these instructions](https://marcc-
hpc.github.io/esc/common/python-environments#conda). 
 
## Relational databases

If you've used a spreadsheet, then you understand how a relational database
works. A *relation* is basically a sheet or table in a spreadsheet which defines
a set of tuples (rows) that all have the same attributes (columns). A relational
database is composed of many such tables along with unique ID numbers for each
row. These IDs allow for relations *between* tables known as a `JOIN` clause.

### An introduction to `sqlite`

In the following exercise we will build a simple database. Our data comes from a
file called `beers_data_simple.txt` (courtesy of [Jan Aerts]([this
exercise](http://vda-lab.github.io/2016/04/mongodb-exercises))). Most of the
following exercise will be completed entirely in BASH and recapped in the steps
below. Note that a `$` indicates the BASH prompt, however most of this exercise
occurs inside the `sqlite3` program, which has its own prompt.

#### Source data

The text in `beers_data_simple.txt` includes the beer name, ABV, and brewery, in
the following format:

```
$ head beers_data_simple.txt
3 Schténg|Brasserie Grain d'Orge|6.0
400|'t Hofbrouwerijke voor Brouwerij Montaigu|5.6
IV Saison|Brasserie de Jandrain-Jandrenouille|6.5
V Cense|Brasserie de Jandrain-Jandrenouille|7.5
VI Wheat|Brasserie de Jandrain-Jandrenouille|6.0
Aardmonnik|De Struise Brouwers|8.0
Aarschotse Bruine|Stadsbrouwerij Aarschot|6.0
Abbay d'Aulne Blonde des Pères 6|Brasserie Val de Sambre|6.0
Abbay d'Aulne Brune des Pères 6|Brasserie Val de Sambre|6.0
Abbay d'Aulne Super Noël 9|Brasserie Val de Sambre|9.0
```

We have used a vertical bar `|` to separate the records in case a comma exists
in any of the names. 
 
#### Building the database

1. Make a new database. This command opens the `SQLite` terminal.

```
$ sqlite3 beers.sqlite3
SQLite version 3.30.1 2019-10-10 20:19:45
Enter ".help" for usage hints.
sqlite>
```

2. Create a table for the breweries. You can extend your output over multiple
lines since they are terminated with a semicolon.

```
sqlite> CREATE TABLE brewery (
    name VARCHAR(128)
);
sqlite> .tables
brewery
sqlite> .schema brewery
```

The command above can repeat the schema back to you.

3. Drop a table.

```
sqlite> DROP TABLE artists;
sqlite> .tables
```

4. Make the table again and include a second table to define the beer and its
alcohol content as a `REAL` type, that is, a floating point number. Note that we
are also including a brewery ID and a "foreign key" which we will explain in
class.

```
sqlite> CREATE TABLE brewery (
    name VARCHAR(128)
);
sqlite> CREATE TABLE beer (
    name VARCHAR(128),
    abv REAL,
    brewery_id INTEGER,
    FOREIGN KEY(brewery_id) REFERENCES brewery(rowid)
);
.schema
```

5. Now we will insert some data and read it back. The `SELECT` command creates a
database "query".

```
sqlite> INSERT INTO brewery VALUES('Dogfishead');
sqlite> INSERT INTO brewery VALUES('Tired Hands');
sqlite> SELECT rowid, name from brewery;
```

6. Now we can insert a beer into the database and associate it with a brewery.

```
sqlite> INSERT INTO beer VALUES ('90 Minute IPA', 9.0, 1);
sqlite> INSERT INTO beer VALUES ('60 Minute IPA', 6.0, 1);
sqlite> INSERT INTO beer VALUES ('HopHands', 5.5, 2);
sqlite> INSERT INTO beer (name) VALUES ('SaisonHands',);
```

7. We can easily search the database for beers from a particular brewery.

```
sqlite> SELECT rowid, name FROM beer WHERE brewery_id=1;
```

8. We can query multiple tables at once using the `INNER JOIN` syntax. In the
following example, we will collect all pairs of brewery names and ABV values by
joining the beer's brewery ID number with the row ID on the brewery table.

```
sqlite> SELECT brewery.name,beer.abv FROM beer INNER JOIN brewery ON
brewery.rowid=beer.brewery_id;
Dogfishead|9.0
Tired Hands|5.5
Dogfishead|6.0
```

If we had created a more complex database, we could use a sequence of `INNER
JOIN` to pull data from multiple tables.

9. We can modify the tables in-place.

```
sqlite> ALTER TABLE beer ADD COLUMN style;
sqlite> UPDATE beer SET style="IPA" WHERE name="90 Minute IPA";
sqlite> select * from beer;
```

The exercise above covers the basics for interacting with `sqlite3` directly in
a terminal. You are welcome to set up a BASH script to ingest and query the
data. For some use-cases, it may be easier to use an interface. In the next
exercise we will examine one type of interface. 
 
### Scripting your `sqlite` workflows

In this example we will repeat some of the work above using `sqlalchemy` from
[this documentation](https://docs.sqlalchemy.org/en/13/orm/tutorial.html). 

**In [2]:**

{% highlight python %}
# clear the existing data if you are starting over
! rm -f beer_alchemy.sqlite beer.sqlite
{% endhighlight %}
 
The `sqlalchemy` library is a front-end for many different types of databases.
The purpose of this library, and others written in different langauges, is to
abstract the programmer from the details of the database. You can use a database
ORM (object-relational mapping) to design a database in your preferred language.
After you work with an ORM for a while, you might choose a different database
**driver** or back-end to suit your performance needs.

In this exercise we will use `sqlalchemy` to design our database directly in
Python and ingest some of the data described in the `sqlite` exercise above.
Since Python makes it somewhat easy to interact with strings, it can reduce the
labor required to ingest the data.

First, we create an "engine" with the following commands. Note that you can use
other drivers in place of `sqlite` below. The following command creates the file
`beer_alchemy.sqlite` on disk, however you are welcome to use `:memory:`
instead. 

**In [3]:**

{% highlight python %}
import sqlalchemy
from sqlalchemy import create_engine
engine = create_engine('sqlite:///beer_alchemy.sqlite',echo=True)
{% endhighlight %}
 
Next we must make a "session" to interact with the ORM. 

**In [4]:**

{% highlight python %}
# make a session instead of transacting directly
from sqlalchemy.orm import sessionmaker
Session = sessionmaker(bind=engine)
session = Session()
{% endhighlight %}
 
We design our database by sub-classing a `delarative_base`. All of our
interactions with the database are abstract, and occur implicitly when we
interact with the library objects (namely `Base`). 

**In [5]:**

{% highlight python %}
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, ForeignKey, Integer, String, Float
from sqlalchemy.orm import relationship
Base = declarative_base()
{% endhighlight %}
 
Now that we have set the stage, we are ready to design our tables, which are
represented as Python classes. One major downside to using an ORM is that this
particular library does not allow for database migrations, hence you must
regenerate the database if you wish to modify it. This imposes an important type
of discipline on your workflow, however all databases must eventually be
migrated, one way or the other. 

**In [6]:**

{% highlight python %}
class Brewery(Base):
    __tablename__ = 'brewery'

    id = Column(Integer, primary_key=True)
    name = Column(String)

    def __repr__(self):
        return f'<Brewery name={self.name}>'
{% endhighlight %}

**In [7]:**

{% highlight python %}
class Beer(Base):
    __tablename__ = 'beer'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    abv = Column(Float)
    brewery_id = Column(Integer, ForeignKey('brewery.id'))
    brewery = relationship('Brewery')

    def __repr__(self):
        return f'<Beer name={self.name}>'
{% endhighlight %}
 
Defining the class is not enough to actually *build* the table. For that, we
must communicate with the engine. Since we asked the engine to echo our results,
we can see how `sqlalchemy` is directly manipulating the database. 

**In [8]:**

{% highlight python %}
Base.metadata.create_all(engine)
{% endhighlight %}

    2019-11-18 20:09:07,168 INFO sqlalchemy.engine.base.Engine SELECT CAST('test plain returns' AS VARCHAR(60)) AS anon_1
    2019-11-18 20:09:07,169 INFO sqlalchemy.engine.base.Engine ()
    2019-11-18 20:09:07,170 INFO sqlalchemy.engine.base.Engine SELECT CAST('test unicode returns' AS VARCHAR(60)) AS anon_1
    2019-11-18 20:09:07,171 INFO sqlalchemy.engine.base.Engine ()
    2019-11-18 20:09:07,173 INFO sqlalchemy.engine.base.Engine PRAGMA main.table_info("brewery")
    2019-11-18 20:09:07,174 INFO sqlalchemy.engine.base.Engine ()
    2019-11-18 20:09:07,176 INFO sqlalchemy.engine.base.Engine PRAGMA temp.table_info("brewery")
    2019-11-18 20:09:07,176 INFO sqlalchemy.engine.base.Engine ()
    2019-11-18 20:09:07,177 INFO sqlalchemy.engine.base.Engine PRAGMA main.table_info("beer")
    2019-11-18 20:09:07,178 INFO sqlalchemy.engine.base.Engine ()
    2019-11-18 20:09:07,180 INFO sqlalchemy.engine.base.Engine PRAGMA temp.table_info("beer")
    2019-11-18 20:09:07,180 INFO sqlalchemy.engine.base.Engine ()
    2019-11-18 20:09:07,182 INFO sqlalchemy.engine.base.Engine 
    CREATE TABLE brewery (
    	id INTEGER NOT NULL, 
    	name VARCHAR, 
    	PRIMARY KEY (id)
    )
    
    
    2019-11-18 20:09:07,183 INFO sqlalchemy.engine.base.Engine ()
    2019-11-18 20:09:07,188 INFO sqlalchemy.engine.base.Engine COMMIT
    2019-11-18 20:09:07,190 INFO sqlalchemy.engine.base.Engine 
    CREATE TABLE beer (
    	id INTEGER NOT NULL, 
    	name VARCHAR, 
    	abv FLOAT, 
    	brewery_id INTEGER, 
    	PRIMARY KEY (id), 
    	FOREIGN KEY(brewery_id) REFERENCES brewery (id)
    )
    
    
    2019-11-18 20:09:07,190 INFO sqlalchemy.engine.base.Engine ()
    2019-11-18 20:09:07,193 INFO sqlalchemy.engine.base.Engine COMMIT

 
We are finally ready to enter some data using the somewhat elegant Python
syntax. 

**In [9]:**

{% highlight python %}
beer = Beer(name='Rt. 113 IPA', abv=7.0)
session.add(beer)
print(beer.id)
{% endhighlight %}

    None

 
The beer has no ID yet because all of our Python objects are not permanently
added to the database until we commit the results. The following command will
finalize this transaction. 

**In [10]:**

{% highlight python %}
session.commit()
{% endhighlight %}

    2019-11-18 20:09:09,577 INFO sqlalchemy.engine.base.Engine BEGIN (implicit)
    2019-11-18 20:09:09,579 INFO sqlalchemy.engine.base.Engine INSERT INTO beer (name, abv, brewery_id) VALUES (?, ?, ?)
    2019-11-18 20:09:09,580 INFO sqlalchemy.engine.base.Engine ('Rt. 113 IPA', 7.0, None)
    2019-11-18 20:09:09,582 INFO sqlalchemy.engine.base.Engine COMMIT


**In [11]:**

{% highlight python %}
# turn off the verbose SQL output
engine.echo = False
{% endhighlight %}
 
Now that this object is entered, we can query the table to return it. 

**In [12]:**

{% highlight python %}
# query the Beer table
session.query(Beer).all()
{% endhighlight %}




    [<Beer name=Rt. 113 IPA>]


 
#### Ingesting data

One reason to move from the `sqlite` terminal or BASH to a higher-level language
like Python is to make it easy to ingest some text. The following example comes
from the top of our `beers.txt` file, which has itself been extracted for this
exercise from a source database. The data below include the beer name, brewery,
and alcohol by volume (ABV). 

**In [13]:**

{% highlight python %}
# start with some source data
data = """
3 Schténg|Brasserie Grain d'Orge|6.0
400|'t Hofbrouwerijke voor Brouwerij Montaigu|5.6
IV Saison|Brasserie de Jandrain-Jandrenouille|6.5
V Cense|Brasserie de Jandrain-Jandrenouille|7.5
VI Wheat|Brasserie de Jandrain-Jandrenouille|6.0
Aardmonnik|De Struise Brouwers|8.0
Aarschotse Bruine|Stadsbrouwerij Aarschot|6.0
Abbay d'Aulne Blonde des Pères 6|Brasserie Val de Sambre|6.0
Abbay d'Aulne Brune des Pères 6|Brasserie Val de Sambre|6.0
Abbay d'Aulne Super Noël 9|Brasserie Val de Sambre|9.0
"""
{% endhighlight %}
 
Let us ingest the data directly from this text file. We will discuss the
following code block in class. In short, we interpret the text file (which has
three columns) and then query the `Brewery` and `Beer` tables, represented by
Python classes. If a particular brewer or beer is missing, we add the objects
and commit. After this step is complete, we can read out all of the rows from
this limited data set. 

**In [14]:**

{% highlight python %}
# ingest the data
for line in data.strip().splitlines():
    beer_name,brewery_name,abv = line.split('|')
    row_brewery = session.query(Brewery).filter_by(name=brewery_name).first()
    if not row_brewery:
        brewery_row = Brewery(name=brewery_name)
        session.add(brewery_row)
        session.commit()
    beer_row = session.query(Beer).filter_by(name=beer_name).first()
    if not beer_row:
        beer_row = Beer(name=beer_name,brewery=brewery_row,abv=abv)
        session.add(beer_row)
        session.commit()
session.query(Beer).all()
{% endhighlight %}




    [<Beer name=Rt. 113 IPA>,
     <Beer name=3 Schténg>,
     <Beer name=400>,
     <Beer name=IV Saison>,
     <Beer name=V Cense>,
     <Beer name=VI Wheat>,
     <Beer name=Aardmonnik>,
     <Beer name=Aarschotse Bruine>,
     <Beer name=Abbay d'Aulne Blonde des Pères 6>,
     <Beer name=Abbay d'Aulne Brune des Pères 6>,
     <Beer name=Abbay d'Aulne Super Noël 9>]


 
Having proved that this method works, we can ingest all of the data from the
source. It's best to package these operations into separate functions. 

**In [18]:**

{% highlight python %}
def ingest_beer(line):
    """Add beer and brewery information into the database."""
    global session,Brewery,Beer
    beer_name,brewery_name,abv = line.split('|')
    # it is important to only grab a single match
    brewery_row = session.query(Brewery).filter_by(
        name=brewery_name).first()
    if not brewery_row:
        brewery_row = Brewery(name=brewery_name)
        session.add(brewery_row)
        session.commit()
    beer_row = session.query(Beer).filter_by(name=beer_name).first()
    if not beer_row:
        beer_row = Beer(name=beer_name,brewery=brewery_row,abv=abv)
        session.add(beer_row)
        session.commit()
{% endhighlight %}

**In [17]:**

{% highlight python %}
with open('beers.txt') as fp:
    text = fp.read().strip().splitlines()
n_records = len(text)
print('we have %d records'%n_records)
{% endhighlight %}

    we have 1691 records


**In [None]:**

{% highlight python %}
# mistakes may later require a rollback
session.rollback()
{% endhighlight %}

**In [19]:**

{% highlight python %}
# easy progress bar in case this takes a while
import sys
def drawProgressBar(f,barLen=20):
    # via https://stackoverflow.com/a/15801617
    sys.stdout.write("\r")
    sys.stdout.write("[{:<{}}] {:.0f}%".format("=" * int(barLen * f), barLen, f*100))
    sys.stdout.flush()
{% endhighlight %}

**In [20]:**

{% highlight python %}
# ingest everything
for lnum,line in enumerate(text):
    drawProgressBar((lnum+1.)/n_records)
    ingest_beer(line)
{% endhighlight %}

    [====================] 100%

**In [21]:**

{% highlight python %}
# query Brewery and Beer together
boozy_breweries = session.query(
    Brewery,Beer).filter(
    Beer.abv>7.0).all()
print('found %d records'%len(boozy_breweries))
{% endhighlight %}

    found 256658 records


**In [23]:**

{% highlight python %}
# something is suspicious
boozy_breweries[1]
{% endhighlight %}




    (<Brewery name='t Hofbrouwerijke voor Brouwerij Montaigu>, <Beer name=V Cense>)


 
The number of records returned from our search above seems absurdly high. This
is an example of an *accidental outer join* which was the result of clumsy code
on my part! Let's try again, and limit our "join" to the desired target: beer. 

**In [24]:**

{% highlight python %}
boozy_breweries = session.query(Brewery).join(
    Beer).filter(
    Beer.abv>9.0).all()
print('found %d records'%len(boozy_breweries))
{% endhighlight %}

    found 71 records

 
This is much more reasonable, particularly since our number of results does not
exceed the number of records. To find the best breweries (i.e. those with the
highest-octane beers), we can *query* both the `Beer` and `Brewery` objects
while joining only on Beer. 

**In [25]:**

{% highlight python %}
boozy_beers_by_brewery = session.query(
    Brewery,Beer).join(Beer).filter(Beer.abv>9.0).all()
print('found %d records'%len(boozy_beers_by_brewery))
{% endhighlight %}

    found 150 records

 
Next we can sort these to find the booziest beers, paired with their brewers. 

**In [29]:**

{% highlight python %}
sort_abv = lambda i:i[1].abv
boozy_ranked = sorted(boozy_beers_by_brewery,key=sort_abv,reverse=True)
for brewer,beer in sorted(boozy_beers_by_brewery,key=sort_abv)[:10]:
    print('%5.2f %s %s'%(beer.abv,beer,brewer))
{% endhighlight %}

     9.20 <Beer name=Keyte-Dobbel-Tripel> <Brewery name=Brouwerij Strubbe>
     9.20 <Beer name=Rochefort 8> <Brewery name=Abdij Notre-Dame de Saint-Rémy>
     9.20 <Beer name=Strandjuttersbier Mong De Vos> <Brewery name=Brouwerij Strubbe voor bierfirma Den Haene>
     9.40 <Beer name=Bière du Corsaire Cuvée Spéciale> <Brewery name=Brouwerij Huyghe>
     9.50 <Beer name=Abbaye des Rocs Grand Cru> <Brewery name=Brasserie de l'Abbaye des Rocs>
     9.50 <Beer name=Achel Blond Extra> <Brewery name=Achelse Kluis>
     9.50 <Beer name=Achel Bruin Extra> <Brewery name=Achelse Kluis>
     9.50 <Beer name=Authentic Triple> <Brewery name=Authentique Brasserie>
     9.50 <Beer name=Bersalis Tripel> <Brewery name=Brouwerij Huyghe voor Brouwerij Oud Beersel>
     9.50 <Beer name=Boerinneken> <Brewery name=De Proefbrouwerij voor Den Ouden Advokaat>

 
Note that you are welcome to unpack and manipulate this data in Python, however
databases are strictly intended to do the heavy lifting for you. Most databases
are useful not only because they allow you to develop complex data structures,
but because their performance far exceeds the in-memory performance of bog
standard Python. 
 
## Unstructured databases: Mongo

For the remainder of this exercise, we will use MongoDB. This database is a
"noSQL" or unstructured or non-relational database. We offer this exercise to
compare its usefulness to the relational databases.

### Raw data

The raw data for the exercise come from the same source as the file above using
these commands on a machine with [Docker](https://docs.docker.com/).

```
docker run -d -p 27017:27017 jandot/mongo-i0u19a
mongoexport --db=i0u19a --collection=beers --out=beers.json
mongoexport --db=i0u19a --collection=breweries --out=brewers.json
docker stop de69c66b8d91 # get the correct name from docker ps
```

The `brewers.json` is provided with this repository and is thanks to [Jan
Aerts](http://vda-lab.github.io/2016/04/mongodb-exercises). 
 
### Starting mongo

In class we will review the use of a `screen` to start Mongo using the following
commands.

```
screen -S mongo
mkdir -p data_mongo
mongod --dbpath=data_mongo
# use the following on Blue Crab to avoid port collisions with other students
mongod --dbpath=data_mongo --port=$(shuf -i8000-9999 -n1)
# note the default port (27017) or a random one via shuffle is required later on
```

If you are using *Blue Crab* you should perform this exercise on an interactive
session using `interact -p express -c 6 -t 120`. Do not forget to load a conda
environment with the dependencies listed at the beginning of this tutorial.

At the end of the exercise we will use the Mongo shell, but for now, we will use
a Python interface. In contrast to `sqlalchemy`, the `pymongo` database is very
similar to the mongo interface itself. 
 
### Ingest the data

Once you have started a mongo daemon in the background, we are readyt o start
using the database. Note that if you use docker or Singularity directly, the
`docker run` command above will make the data available automatically. 

**In [30]:**

{% highlight python %}
from pymongo import MongoClient
# if you are using a random port, substitute it below
client = MongoClient('localhost', 27017)
{% endhighlight %}
 
To complete the exercise we must create a database and a collection. 

**In [None]:**

{% highlight python %}
db = client["beer_survey"]
beers = db["beers"]
{% endhighlight %}
 
First, we should take a look at the raw data which we exported for this
exercise. 

**In [31]:**

{% highlight python %}
! head -n 3 beers.json
{% endhighlight %}

    {"_id":{"$oid":"5dd300d16881a20dc8b96777"},"beer":"3 Schténg","brewery":"Brasserie Grain d'Orge","type":["hoge gisting"],"alcoholpercentage":6.0}
    {"_id":{"$oid":"5dd300d16881a20dc8b96778"},"beer":"IV Saison","brewery":"Brasserie de Jandrain-Jandrenouille","type":["saison"],"alcoholpercentage":6.5}
    {"_id":{"$oid":"5dd300d16881a20dc8b96779"},"beer":"V Cense","brewery":"Brasserie de Jandrain-Jandrenouille","type":["hoge gisting","special belge"],"alcoholpercentage":7.5}

 
We can see that each JSON entry includes a beer and "type" along with the ABV.
The JSON syntax is a useful way to represented a nested dictionary or tree of
values. 

**In [34]:**

{% highlight python %}
import json,pprint
{% endhighlight %}

**In [35]:**

{% highlight python %}
# if you repeat this exercise, it may be useful to clear the database
try: beers.delete_many({})
except: pass
{% endhighlight %}
 
Next we will unpack each line with JSON and use `insert_one` to directly add
them to the database. Since  this database is unstructured, we do not have to
define the schema ahead of time. 

**In [36]:**

{% highlight python %}
with open('beers.json') as f:
    for line in f.readlines():
        entry = json.loads(line)
        beers.insert_one(dict(
            beer=entry['beer'],
            abv=entry['alcoholpercentage'],
            brewery=entry['brewery'],
            style=entry['type'],
        ))
{% endhighlight %}

**In [37]:**

{% highlight python %}
# we can count the records
beers.count_documents({})
{% endhighlight %}




    1691


 
The syntax is slightly different, but as with the relational database, queries
are relatively easy to write. 

**In [39]:**

{% highlight python %}
boozy = list(beers.find({"abv": {"$gt": 8}}))
print("there are %d breweries with boozy beers"%
    len(list(set([i['brewery'] for i in boozy]))))
{% endhighlight %}

    there are 168 breweries with boozy beers

 
We can sort the results of our query directly in Python. 

**In [47]:**

{% highlight python %}
[(i['beer'],i['abv']) 
    for i in sorted(beers.find({"abv": {"$gt": 8}}),
    key=lambda x:x['abv'],reverse=True)[:10]]
{% endhighlight %}




    [('Black Damnation V (Double Black)', 26.0),
     ("Cuvée d'Erpigny", 15.0),
     ('Black Albert', 13.0),
     ('Black Damnation I', 13.0),
     ('Black Damnation III (Black Mes)', 13.0),
     ('Black Damnation IV (Coffée Club)', 13.0),
     ('Bush de Noël Premium', 13.0),
     ('Bush de Nuits', 13.0),
     ('Bush Prestige', 13.0),
     ('Cuvée Delphine', 13.0)]


 
Alternately, we can chain together a series of queries to filter the data. In
the following example we find all beers above 8% ABV, group them by brewery,
take the average, and then collect a sample of five. 

**In [62]:**

{% highlight python %}
agg = db.beers.aggregate([
    {"$match": {"abv": {"$gt": 8}}},
    {"$group": {"_id": "$brewery", "avg": {"$avg": "$abv"}}},
    {"$sample": {"size": 5}}
    ])
print('\n'.join(['%05.2f %s'%(i['avg'],i['_id']) for i in agg]))
{% endhighlight %}

    08.50 Brasserie Millevertus
    08.50 Brouwersverzet bij Brouwerij Gulden Spoor
    08.50 Alken-Maes (Heineken)
    08.40 Brouwerij Strubbe voor Brouwerij Crombé
    08.45 AB InBev

 
The syntax above, including the use of `$group` and `$match` is reviewed in the
list of *accumulators*
[here](https://docs.mongodb.org/manual/reference/operator/aggregation-
pipeline/). 
 
### Review the Mongo shell

Before continuing, it is useful to check on the database directly with `mongo`.
The commands above can also be completed directly inside Mongo using some of the
following commands. Make sure your daemon is running first.

```
mongo --host localhost
show dbs
use beer_survey
db.beers.findOne()
db.beers.aggregate([
  {$match: {abv: {$gt: 8}}},
  {$group: {_id: "$brewery", avg: {$avg: "$abv"}}},
  {$sample: {size: 5}}
])
```

When designing your calculations, you should choose the combination of database
and ORM or interface that makes it easiest to manipulate your data. 
 
### Advanced filtering

In our final example, we will use *map reduce* to count the number of beers per
brewery. This requires some JavaScript implemented with the `bson` library,
since this is Mongo's native language. 

**In [87]:**

{% highlight python %}
from bson.code import Code
{% endhighlight %}

**In [88]:**

{% highlight python %}
mapped = Code("""
function () { 
    emit(this.brewery, 1);
}
""")
{% endhighlight %}

**In [89]:**

{% highlight python %}
reduce = Code("""
function (brewery, values) {
    return Array.sum(values)
}
""")
{% endhighlight %}

**In [90]:**

{% highlight python %}
result = db.beers.map_reduce(mapped,reduce,out="numberBeersPerBrewery")
result_ranked = sorted(result.find(),key=lambda x:x['value'],reverse=True)
result_ranked[:5]
{% endhighlight %}




    [{'_id': 'Brouwerij Huyghe', 'value': 43.0},
     {'_id': 'Brouwerij Van Honsebrouck', 'value': 36.0},
     {'_id': 'Brouwerij Van Steenberge', 'value': 32.0},
     {'_id': 'Brouwerij De Regenboog', 'value': 31.0},
     {'_id': 'Brouwerij Alvinne', 'value': 30.0}]


 
The map-reduce framework is a general one which can be used in many other
contexts. 
 
That's all for today! This exercise only scratched the surface. Databases are
extremely useful tools that help to extend your programs to accomodate larger
and more complex data structures. 

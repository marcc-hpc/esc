
# imports for plotting
import matplotlib as mpl
# turn off the backend
mpl.use('Agg')
import matplotlib.pyplot as plt

import glob
fns = glob.glob('result_*')

# collect all of the data
collected = {}
for fn in fns:
    with open(fn) as fp:
        text = fp.read()
    lines = text.splitlines()
    # join and split the data
    lines_reduced = dict([(i.split()[0],' '.join(i.split()[1:])) for i in lines])
    # index the data by beta
    collected[lines_reduced['beta']] = lines_reduced

# reformulate the results
ts = [(float(j['beta']),float(j['average'])) for i,j in collected.items()]
ts = sorted(ts,key=lambda x:x[0])

# make a plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(*zip(*ts),'-',label='susceptible')
ax.set_xlabel('beta')
ax.set_ylabel('susceptible individuals')
plt.savefig('beta_susceptible.png')
plt.close()

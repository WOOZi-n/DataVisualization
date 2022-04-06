import pandas as pd
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

x = np.linspace(0, 2*pi, 30)
y1 = np.sin(x)
dat = pd.DataFrame({'x' : x, 'y1' : y1})
dat.plot.scatter('x','y1')

fig,ax = plt.subplots()
dat.plot.scatter(x = 'x', y = 'y1', ax = ax)
dat.plot.scatter(x = 'y1', y = 'x', ax= ax)

mtcars = sm.datasets.get_rdataset("mtcars", 'datasets', cache = False)
mtcars_dat = mtcars.data.copy()
mtcars_dat.head()

mpg = mtcars_dat["mpg"]
cyl = mtcars_dat["cyl"]


plt.scatter(cyl, mpg)
plt.xlabel("cyl")
plt.ylabel("mpg")

dat = pd.DataFrame({'cyl' : cyl , "mpg" : mpg})
dat.plot.scatter('cyl', 'mpg')

len(mtcars_dat.index)
mtcars_dat["company"] = [s.split()[0] for s in mtcars_dat.index]
group_dat = mtcars_dat.groupby('company')
group_dat.groups
fig , ax = plt.subplots()
for name, group in group_dat:
    print(name)
    print(group)
    
for name, group in group_dat:
    ax.plot(group.mpg, group.disp , 'o', label = name)
ax.legend(loc = 'lower center', ncol = 11, bbox_to_anchor = (0.4, -0.4))


mtcars_dat['company'] = pd.Categorical(mtcars_dat['company'])
mtcars_dat.plot.scatter(x = 'mpg', y= 'disp', c= 'company', cmap = 'viridis')


sns.scatterplot(x = 'mpg', y = 'disp', hue = 'company', data= mtcars_dat)
sns.pa



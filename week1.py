import matplotlib.pyplot as plt
import numpy as np
from math import pi
import pandas as pd
import statsmodels.api as sm
import seaborn as sns


x = np.linspace(0, 2*pi, 30)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, 'o-r')
plt.plot(x, y2, 'o')
plt.plot(x,y1,'o',x,y2, 'o')

plt.scatter(x, y1, s = 100, c = 'red')

dat = pd.DataFrame({"x": x, "y1" : y1, "y2" : y2})
dat.plot.scatter(x = 'x', y = 'y1')
dat.plot.scatter(x = 'x', y = 'y2')

fig, axes = plt.subplots()
dat.plot.scatter(x = 'x', y= 'y1', ax = axes)
dat.plot.scatter(x = 'x', y = 'y2', ax =axes)

mtcars = sm.datasets.get_rdataset("mtcars", "datasets", cache = False)
mtcars_dat = mtcars.data.copy()
mtcars_dat

plt.plot(mtcars_dat["mpg"], mtcars_dat["disp"], "o")
plt.scatter(mtcars_dat["mpg"], mtcars_dat["disp"])
plt.xlabel("mpg")
plt.ylabel("disp")
plt.title("mpg and disp")

s=mtcars_dat.index
s.split()
print(s)

mtcars_dat["company"] = [s.split()[0] for s in mtcars_dat.index]
group_dat = mtcars_dat.groupby("company")
fig, ax = plt.subplots()
for name, group in group_dat:
    ax.scatter(group.mpg, group.disp, label = name)
axes.legend(loc = "lower center",
          ncol = 11,
          bbox_to_anchor = (0.5, -0.5))

for name,group in group_dat:
    ax.plot(group.mpg, group.disp, 
            'o', label= name)
 
    
fig, axes = plt.subplots()
for name, group in group_dat:
    axes.scatter(group.mpg, group.disp, label = name)
axes.legend(loc = 'lower right', ncol = 4, bbox_to_anchor = (0.5, 0.5, 1, 2))



mtcars_dat2 = mtcars_dat.copy()
mtcars_dat2['company'] = pd.Categorical(mtcars_dat2["company"])
mtcars_dat2.plot.scatter(x = 'mpg', y = 'disp', c = 'company', cmap = 'viridis')


viridis = plt.get_cmap("viridis")
viridis
viridis(1)
len(viridis.colors)

sns.scatterplot(x = "mpg", y = "disp", hue = "company", s = 30, data = mtcars_dat2)

wt_mean = mtcars_dat.groupby('company').wt.transform("mean")

wt_mean2 = mtcars_dat.groupby('company')["wt"].mean()

print([wt_mean2, wt_mean])

sns.pairplot(mtcars_dat[["mpg", "cyl", "disp","hp","drat","wt"]], diag_kind = "kde", hue = 'company', corner = True)


x = np.linspace(0, 2*pi, 30)
y2 = np.cos(x) + 0.1 * np.random.randn(len(x))
plt.plot(x, y2, 'o')
plt.axhline(y =1, linestyle = 'dashed')


x = mtcars_dat["mpg"]
y = mtcars_dat["disp"]
X = sm.add_constant(x)
lin_model = sm.OLS(y,X)
res = lin_model.fit()
res.summary()

plt.plot(x,y, "o")
plt.axline(xy1 = (0,res.params[0]), slope = res.params[1], c = 'red')

plt.title("intercept : {}, slope : {})".format(*res.params.round(1)))






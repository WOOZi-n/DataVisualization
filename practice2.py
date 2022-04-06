import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

mtcars = sm.datasets.get_rdataset("mtcars", "datasets")
mtcars_dat = mtcars.data.copy()
mtcars_dat.head()
mtcars_dat['company'] = [s.split()[0] for s in mtcars_dat.index]
mtcars_dat['company']
df = mtcars_dat
plt.axis([0,50,0,500])

group_dat = mtcars_dat.groupby('company')

fig,ax = plt.subplots()
for name, group in group_dat:
    ax.scatter(group.mpg, group.disp, label = name)
ax.legend(loc = 'lower center', ncol = 11, bbox_to_anchor = (0.5, -0.5))


sns.pairplot(mtcars_dat)

x = mtcars_dat["mpg"]
y = mtcars_dat['disp']
x = sm.add_constant(x)
lin_model = sm.OLS(y,x)
res = lin_model.fit()
res.summary()
res.params[0]

plt.plot(x['mpg'], y , 'o')
plt.axline((0, res.params[0]), slope = res.params[1], c= 'red')


state_region = sm.datasets.get_rdataset("usa_states", 'stevedata')
state = state_region.data.copy()
count = state.region.value_counts()
count
plt.bar(x = count.index, height = count.values, color = ['black', 'red', 'yellow' ,'green'])


count.plot.bar(y = 'region', color = ['black','red','green', 'blue'])




# t-test between Species

iris= sm.datasets.get_rdataset("iris", "datasets")    
iris_dat = iris.data.copy()
iris_dat.head()
grouped_iris = iris_dat.groupby("Species")






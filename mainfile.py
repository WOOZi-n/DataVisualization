#녹강 듣자. 내용이 좀 다르다

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

#막대그래프
#1. matplotlib.pyplot.bar함수

state_region = sm.datasets.get_rdataset("usa_states", 'stevedata') #stevedata 내의 usa_states를 불러옴

state = state_region.data.copy()
count = state.region.value_counts()

plt.bar(x = count.index, height = count.values, color = ['black', 'red', 'green', 'blue'])
plt.ylabel('Freq') #label이 생기는 plot은 어디?

plt.xticks(ticks = range(4), labels = [s.lower() for s in count.index])
#ticks와 label의 수가 같을 때 x축을 네 범주로 나누며, 각각에 label을 대응한다

#2. pandas.DataFrmae.plot.bar

count.plot.bar(y = 'region', color = ['black', 'red', 'green', 'yellow'])
#mtcars 데이터 불러오기
mtcars = sm.datasets.get_rdataset("mtcars", "datasets")
mtcars_dat = mtcars.data.copy()
mtcars_dat.columns
mtcars_dat['company'] = [s.split()[0] for s in mtcars_dat.index]
mtcars_dat.groupby("company")[["cyl", "wt"]].mean().plot.bar() #company에 따른 cyl,wt의 평균을 표현
mtcars_dat.groupby("company")[["cyl", "wt"]].mean().plot.bar(subplots = True, rot = 45) 
#subplots : 각기다른 플롯으료 표현, 값 index rotate

#3. pyplot으로 방금과 같은 그래프그리기
group_dat = mtcars_dat.groupby('company')[["cyl", "wt"]].mean()
print(group_dat)
group_dat.shape #dim이랑 같은거
fig = plt.figure() #도화지
ax1 = fig.add_subplot(211) #축의 위치, figure 내에서의 위치표현 (nrows = 2, ncol = 1, index = 1)
ax2 = fig.add_subplot(212)
# ax = plt.subplots(nrow = 2)
# ax1,ax2 = ax[0], ax[1]
ax1.bar(x = range(group_dat.shape[0]), height = group_dat.cyl, color = 'red')
ax2.bar(x = range(group_dat.shape[0]), height = group_dat.wt, color = 'blue')
ax1.set_xticks(range(group_dat.shape[0]))
ax1.set_xticklabels(group_dat.index)
ax1.tick_params(rotation =45)
ax2.set_xticklabels(group_dat.index)
ax2.set_xticks(range(group_dat.shape[0]))
ax2.tick_params(rotation =45)

#side_by_side bar plot
fig = plt.figure()
ax = fig.add_subplot(111)
width = 0.3
ind = np.arange(group_dat.shape[0])
bar1 = ax.bar(x = ind, height = group_dat.cyl, width = width, color = 'red')
bar2 = ax.bar(x = ind+width, height = group_dat.wt, width = width , color = 'blue') #굵기만큼 옆으로 밀어주기
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(group_dat.index)
ax.tick_params(rotation = 45)
ax.legend((bar1[0], bar2[0]), group_dat.columns)  # 어케했누?

#4. seaborn 라이브러리의 barplot 함수 사용
count
count.shape
type(count)  # count는 series 형태
count.reset_index(drop = False) #index를 0,1,2,로 되돌림(default일때) )> dataframe형태로
sns.barplot(x = 'index', y= 'region', data = count.reset_index(drop = False)) 
barplot_dat= group_dat.reset_index(drop = False).melt(id_vars = 'company') #  melt : 행 녹이기,  id_vars : 기준 컬럼

#ㅈ wide form
sns.barplot(data = group_dat, ci = 95) # ci : confidence interval
#long form
sns.barplot(x = 'company', y = 'value', hue = 'variable', data= barplot_dat)



#실습 : iris 데이터를 이용한 막대그래프 및 t 검정

iris = sm.datasets.get_rdataset("iris", "datasets", cache = False)
iris_dat = iris.data.copy()
iris_melt = iris_dat.melt(id_vars = 'Species')
sns.barplot(x = 'variable', y = 'value', hue = 'Species', data= iris_melt, ci = 95)



# using pyplot

se = lambda x : x.std() / np.sqrt(x.shape[0]) # 시그마/루트n : 표본평균
iris_mean = iris_dat.melt(id_vars = 'Species').groupby(by = ['Species', 'variable']).agg(
    {'value' : ['mean', se]}).reset_index(drop = False).sort_values(['variable', 'Species'])
#agg함수 : 데이터에 부분적을 함수적용하는 메서드 여기선 value 열에 함수를 적용하여 반환
#reset_index를 안하면 인덱스가 integer로 나오지 않는다 ? 왜 인덱스가 사라지지? <- groupby 사용하면 기본적으로 그룹 라벨이 인덱스가 된다
iris_mean.rename(columns = {'<lambda_0>' : 'se'}, inplace = True)
iris_mean
fig = plt.figure()
ax = fig.add_subplot(111)
width1 = 0.3; width2 = 1; n = iris_mean.shape[0] # n : iris species x variable 순서쌍 수 : 12
n_type = iris_mean.variable.unique().shape[0] #variable 수
n_spe = iris_mean.Species.unique().shape[0]  #species 수
grid = [0 + width1 * i for i in range(n)]  # [0 0.3 0.6 0.9 ---- 3.3]
grid = grid + np.repeat(range(1, n_type + 1), n_spe) #[1,1,1,2,2,2,3,3,3,4,4,4]
color = ['red', 'blue','green'] * n_type
bar = ax.bar(x = grid, height = iris_mean.value['mean'], width = width1 , color = color)
#해석해봐,,,

error = ax.errorbar(x = grid, y = iris_mean.value['mean'], yerr = iris_mean.value['se'], fmt = " ", color = 'black')
ax.set_xticks([np.mean(grid[(3*i) : (3*i + n_spe)]) for i in range(4)]) 
ax.set_xticklabels(iris_mean.variable.unique())
ax.legend((bar[0], bar[1], bar[2]), iris_mean.Species.unique())


# 각 변수별 Species간의 t 검정을 시행해보자

grouped_iris = iris_dat.groupby("Species")
group = list(grouped_iris.groups.keys())[0] # groupby 객체는 groups라는 딕셔너리에 뭐가 어느그룹에 들어있는지를 표시해줌
temp_dat = grouped_iris.get_group(group) #  이 그룹에 들어있는 거 가져옴
temp_dat.columns
from itertools import permutations
comb = list(permutations(
    [s for s in temp_dat.columns if s != 'Species'],2)) # 특성 중 2개 선택하는 순열
for i in range(len(comb)):
    t_dat = temp_dat.loc[:,
                         temp_dat.columns.isin(comb[i])] #comb 내의 2개 특성 데이터 추출
    t_val, p_val, degree = sm.stats.ttest_ind(
        x1 = t_dat.iloc[:,0],
        x2 = t_dat.iloc[:,1],
        alternative = 'two-sided',
        usevar = 'unequal')  # t-test 해주는 메서드
    
    print(" vs ".join(comb[i]))
    print('T-value : ', t_val)
    print('P-value : ', p_val)
    print("-"*50)
    
    
    
#pie chart by pyplot

usa = sm.datasets.get_rdataset('usa_states', 'stevedata', cache = False)
usa_states = usa.data.copy()
usa_states
counts = usa_states.region.value_counts()
plt.pie(counts, labels = counts.index, autopct = "%.1f%%")

#pie chart by pandas     plot.pie method

counts.plot.pie(autopct = "%.1f%%",
                shadow = True,
                explode = [0.05]*4) # 소수점 한자리만, EXPLODE : 조각간 퍼진 간격



#boxplot

x = np.random.randn(100)
plt.boxplot(x = x, labels = 'x')
plt.boxplot(iris_dat.drop("Species", axis = 1), 
            labels = iris_dat.columns.drop("Species"),
            vert = False) # vert : 가로세로 결정

box = plt.boxplot(iris_dat.drop("Species", axis = 1), 
            labels = iris_dat.columns.drop("Species"),
            patch_artist = True) # patch_artist : 색깔관련

color = ['red', 'blue','green','yellow']
for b, col in zip(box["boxes"], color):
    b.set_facecolor(col) # 색깔설정메서드
    
#boxplot by pandas

iris_dat.boxplot()
#그룹별 boxplot
iris_dat.boxplot(column = ["Sepal.Length", "Petal.Length"], by = 'Species') # column 마다 종류에 따른 boxplot 도출

box = iris_dat.boxplot(column = ["Sepal.Length", "Petal.Length"], by = 'Species',
                       patch_artist = True, return_type = 'dict') #return_type?

box.iteritems()

for var, dic in box.iteritems():
    for b, col in zip(dic["boxes"], color):
        b.set_facecolor(col)

iris_melt = iris_dat.melt(id_vars = 'Species')
sns.boxplot(x = 'variable', y= 'value', hue = 'Species', data = iris_melt)





#220407수업

# 박스플롯에 접근하기
box = iris_dat.boxplot(column = ['Sepal.Length', 'Petal.Length'],
                       by  ='Species',
                       patch_artist = True,
                       return_type = 'dict')

box[0]
box[0].keys()
box[0]['medians'][0].set_color('red')
box[0]['fliers'][2].set_markeredgewidth(10)
box[0]['caps'][0].set_color('red')


#histogram
#using matplotlib.pyplot
plt.hist(iris_dat["Sepal.Length"])
n, bins, patches = plt.hist(iris_dat["Sepal.Length"], bins = 31)
#bins? : 

n # 각 계급 간 도수 bins : 계급들
x = iris_dat["Sepal.Length"]
(iris_dat["Sepal.Length"].max() - iris_dat["Sepal.Length"].min())/31
# 한 구간의 길이

breaks = np.linspace(x.min(), x.max(), 50)
breaks
# bins에 array 대입
plt.hist(iris_dat["Sepal.Length"], bins = breaks)


n, bins, patches = plt.hist(iris_dat["Sepal.Length"], density = True)
# 히스토그램의 넓이가 1이 되도록 해준다(도수가 아니라 비율표시)

np.diff(bins)[0]*n[0] # 연속형 변수 확률구하기



#높이가 비율이 됨 (?)
plt.hist(iris_dat["Sepal.Length"], 
         weights = np.repeat(1/iris_dat.shape[0],
                             iris_dat.shape[0]))

#그룹별 히스토그램
grouped_iris = iris_dat.groupby("Species")
list(grouped_iris)
keys = grouped_iris.groups.keys()
for key in keys:
    dat = grouped_iris.get_group(key)
    plt.hist(dat["Sepal.Length"], 
             alpha= 0.5, label = key)
plt.legend(loc = 'upper left')

#using pandas

iris_dat.hist()
iris_dat.hist(column = 'Sepal.Length', by = 'Species')

#using seaborn

sns.histplot(data= iris_dat, x = 'Sepal.Length', hue = 'Species',
             stat = 'density')

fig = plt.figure(figsize = (3,3))
# axes = fig.add_subplot(111)
ax1 = fig.add_subplot(4,1,1)
ax2 = fig.add_subplot(2,1,2)
ax3 = fig.add_subplot(3,3,8)
ax1.hist(iris_dat["Sepal.Length"])
ax2.hist(iris_dat["Sepal.Length"])
ax3.boxplot(iris_dat["Sepal.Length"])

#legends
fig, axes = plt.subplots(1,2)
axes[0].hist(iris_dat["Sepal.Length"])
axes[1].hist(iris_dat["Sepal.Width"])
#axes[0].legend("Hi")
#axes[1].legend("bye")
#차이점?
plt.legend(["Hi"]) # 가장 최근에 그려진 그래프에 범례 생성
fig.legend(["Hello"]) # 도화지에 대한 범례

fig,axes = plt.subplots(1,2)
axes[0].hist(iris_dat["Sepal.Length"], alpha = 0.5)
axes[0].hist(iris_dat["Sepal.Width"], alpha = 0.5)
axes[1].hist(iris_dat["Petal.Length"], alpha = 0.5)
axes[1].hist(iris_dat["Petal.Width"], alpha = 0.5)

axes[0].legend(["Sepal.Length", "Sepal.Width"])
axes[1].legend(["Petal.Length", "Petal.Width"])
#그려진 순서대로 적용

fig.legend(["PLT1_1", "PLT1_2", "PLT2_1", "PLT2_2"])

fig,axes = plt.subplots(1,1)
axes.hist(iris_dat["Sepal.Length"], alpha = 0.5, label = "Sepal.Length")
axes.legend(title = 'IRIS', 
            loc = 'upper left',
            fontsize = 20,
            title_fontsize = 30)
#label 이 있으면 legend 메서드가 자동으로 이름을 추론하여 적용

states = sm.datasets.get_rdataset("usa_states", "stevedata")
state_dat = states.data.copy()
count = state_dat.region.value_counts()

fig, axes = plt.subplots(1,1)
axes.bar(x = count.index, height = count)
plt.xlabel("Region")
plt.ylabel("Count")

fig, axes = plt.subplots(1,2)
axes[0].bar(x = count.index, height = count)
axes[1].pie(x = count.values, labels = count.index)
axes[0].set_xlabel("Region",
                   fontsize = 10,
                   color = 'red',
                   fontweight = "bold")
axes[0].set_ylabel("Count" ,
                   fontstyle = 'italic')
axes[1].set_xlabel("pie plot")

# AXIS  범위설정

x = np.linspace(0,2*np.pi, 30)
y = np.cos(x)
fig, axes  = plt.subplots(1,1)
axes.plot(x,y)
plt.xlim(0, np.pi)
plt.ylim(-1, 1)

fig, axes = plt.subplots(1,1)
axes.plot(x,y)
plt.xticks(ticks = [0,3,6],
           labels = [10,20,30]) # 표시되는 ticks

axes.tick_params(axis = 'both', color = 'red', 
                 labelcolor = 'blue',
                 width = 3,
                 direction = 'inout') # x,y,both


#3차원 플롯

mtcars = sm.datasets.get_rdataset("mtcars", "datasets")
mtcars_dat = mtcars.data.copy()

fig = plt.figure()
axes = fig.add_subplot(1,1,1, projection = '3d') # 3차원 축 형성

# fig, axes = fig.subplots(1,1, subplot_kw = dict(projection = '3d'))

axes.scatter(mtcars_dat.wt, mtcars_dat.disp, mtcars_dat.mpg,
             depthshade = False)
#자동적으로 원근감 적용 : depth를 alpha로 표현

axes.scatter(mtcars_dat.wt,
             mtcars_dat.mpg,
             zdir = 'y', #y 축 고정
             zs = mtcars_dat.disp.max(),
             depthshade = False)

axes.scatter(mtcars_dat.disp,
             mtcars_dat.mpg,
             zdir = 'x', #y 축 고정
             zs = mtcars_dat.wt.min(),
             depthshade = False)

axes.scatter(mtcars_dat.wt,
             mtcars_dat.disp,
             zdir = 'z', #y 축 고정
             zs = mtcars_dat.mpg.min(),
             depthshade = False)


axes.set_xlim([mtcars_dat.wt.min(), mtcars_dat.wt.max()])
axes.set_ylim([mtcars_dat.disp.min(), mtcars_dat.disp.max()])
axes.set_zlim([mtcars_dat.mpg.min(), mtcars_dat.mpg.max()])

axes.set_xlabel('wt')
axes.set_xlabel('disp')
axes.set_xlabel('mpg')
axes.view_init(90, 0) # 바라보는 각도. 위, 옆


x = np.linspace(-4*np.pi, 4*np.pi, 50)
y = np.linspace(-4*np.pi, 4*np.pi, 50)
z =  x**2 + y**2

fig = plt.figure()
axes = fig.add_subplot(111, projection = '3d')
axes.plot(x,y,z)



# 3d surface

volcano = sm.datasets.get_rdataset('volcano', 'datasets').data.copy()
volcano.shape

x = np.arange(1, volcano.shape[0] + 1)
y = np.arange(1, volcano.shape[1] + 1)
xx, yy = np.meshgrid(x,y)
xx.shape
yy.shape

fig = plt.figure()
axes = fig.add_subplot(111, projection = '3d')
axes.plot_surface(xx.T, yy.T, volcano.values)
# transpose 시키는이유?


#interative plots

import plotly.express as px
import plotly.io as pio

pio.renderers.default = 'browser'
# 그래프를 브라우저에 띄움

mtcars_dat.cyl = pd.Categorical(mtcars_dat.cyl)
fig = px.scatter(data_frame = mtcars_dat, 
           x = "wt",
           y = "mpg",
           color = 'cyl',
           symbol = 'cyl')

fig.show()



import os
os.chdir("C:\Coding\python\data")

passengers = pd.read_csv("C:\Coding\python\data\AirPassengers.csv")
passengers.head()


passengers.Month = pd.to_datetime(passengers.Month,
               format = "%Y-%m")
passengers["cumsum"] = passengers.Passengers.cumsum() / 50
passenger_melt = passengers.melt(id_vars = "Month")
#두 변수를 동시에 그리기 위해 melting 필요
fig = px.line(passenger_melt, 
              x = "Month",
              y = "value",
              color = "variable",
              labels = {'variable' : 'type',
                        'cumsum' : 'passenger(fifty)'}
              )
#cumsum 안바뀌네 ?
fig.show()

mtcars_dat
fig = px.scatter_3d(mtcars_dat,
                    x = 'wt',
                    y = 'disp',
                    z = 'mpg',
                    color = 'cyl',
                    symbol = 'cyl')
fig.show()

#using plotly.graph_objects

import plotly.graph_objects as go
obj = go.Scatter3d(x = mtcars_dat.wt,
                   y = mtcars_dat.disp,
                   z = mtcars_dat.mpg,
                  mode = 'markers')
fig = go.Figure(data = obj)
fig.show()


# surface 그래프 그리기
x
y
obj = go.Surface(x=x, y=y,
           z = volcano.values)
fig = go.Figure(obj)
fig.show()

#회귀분석!
simul = pd.read_csv("C:\Coding\python\data\simulation.csv")
simul.head()

obj = go.Scatter3d(x = simul.x1, y = simul.x2,
                   z = simul.y,
                   mode = 'markers')
fig = go.Figure(obj)
fig.show()

X1 = sm.add_constant(simul["x1"])
lm1 = sm.OLS(simul.y, X1)
lm1.fit().summary()

X2 = sm.add_constant(simul["x2"])
lm2 = sm.OLS(simul.y, X2)
lm2.fit().summary()

X = sm.add_constant(simul[["x1", "x2"]])
lm = sm.OLS(simul.y, X)
lm.fit().summary()

#데이터 불러오

import pandas as pd
import os 


#dau

dau = pd.read_table(r"C:\Coding\python\game_user_data\dau\game-01\2013-05-01\data.tsv")

list = os.listdir(r"C:\Coding\python\game_user_data\dau\game-01")
for i in list[1:]:
    data = pd.read_table(r"C:\Coding\python\game_user_data\dau\game-01\{}\data.tsv".format(i))
    dau = pd.concat([dau, data], axis = 0)
    
dau = dau.reset_index(drop = True)


#dpu

list2 = os.listdir(r"C:\Coding\python\game_user_data\dpu\game-01")

dpu = pd.read_table(r"C:\Coding\python\game_user_data\dpu\game-01\2013-05-01\data.tsv")

for i in list2[1:]:
    data = pd.read_table(r"C:\Coding\python\game_user_data\dpu\game-01\{}\data.tsv".format(i))
    dpu = pd.concat([dpu, data], axis = 0)
    
dpu = dpu.reset_index(drop = True)

dpu.head(10)
type(dpu)

dpu.head(35)


#수업내용
data = pd.merge(dau, dpu, on = ["log_date", "user_id","app_name"], how = 'left')
data.loc[data.payment.isna(), "payment"] = 0
data.head()

#해당기간동안 게임 한번이상 접속유저
data.user_id.unique().shape

#월별 접속자 수?
data.log_date = pd.to_datetime(data.log_date, format = "%Y-%m-%d")
data['month'] = data.log_date.dt.month
data.log_date.dt.year
data.log_date.dt.day

data.log_date + pd.Timedelta(weeks = 3)
data.log_date + pd.Timedelta(months = 3)
#month year은 제공안됨
#방법
from dateutil.relativedelta import relativedelta
data.log_date[0] + relativedelta(months = 1)
data.log_date[0] + relativedelta(years = 1)
#iterable 객체계산이 안됨. 하나의 데이터만 처리가능


data.groupby("month").apply(lambda x : len(x.user_id.unique()))
#apply는 행별로 적용(groupby 객체임 고려) / x 가 groupby 객체가 됨
#apply 안쓰면...어질어질하다
data.groupby("month").user_id.unique()


data.groupby("month").payment.mean()
data.groupby("month").payment.max()
data.groupby("month").payment.agg(avg = 'mean', maximum = 'max')
data.groupby("month").payment.agg(avg = np.mean, maximum = 'max')

data["log_month"] = data.groupby(
    ["month","user_id"]).user_id.transform(
        lambda x : len(x))
        
data.groupby(["month",'user_id']).user_id
        
data.head()

log_vs_pay = data.groupby(
    'user_id').agg(
        {'log_month' : lambda x : str(np.where((x > 15).all() , "high",
                                           np.where((x > 10).all(), 'mid', 'low'))),
         'payment' : 'mean'})

        
log_vs_pay


pay_group = pd.cut(log_vs_pay.payment,
       bins = [0, 50, 100, np.inf],
       right = False,
       labels = ['low', 'mid','high'])

log_vs_pay["pay_group"] = pay_group
        
ct = pd.crosstab(log_vs_pay.log_month,
                 log_vs_pay.pay_group)        
#표에서 범주 순서 맞추기

log_vs_pay.log_month = pd.Categorical(log_vs_pay.log_month,
                                      categories = ['low','mid','high'])        

log_vs_pay.pay_group = pd.Categorical(log_vs_pay.pay_group,
                                      categories = ['low','mid','high'])        


cp = ct.sum(axis = 0) / ct.values.sum()
rp = ct.sum(axis = 1) / ct.values.sum()
et = ct.values.sum()*np.outer(rp,cp)
(ct-et) / np.sqrt(et) # n-u table

# 독립성 검정
from scipy.stats import chi2_contingency

diff = (ct-et) / np.sqrt(et)
diff_mat = diff.copy()

diff_mat["log_month"] = diff.index
diff_melt = diff_mat.melt(id_vars = 'log_month',
              value_vars = ['low', 'mid','high'],
              var_name = 'group',
              value_name = 'count')

diff_melt['x'] = diff_melt.log_month.astype(str) + '=' \
    + diff_melt.group

import seaborn as sns
axes = sns.barplot(x = 'x', y = 'count', data = diff_melt)
axes.tick_params(axis = 'x',rotation = 90)
for patch in axes.patches:
    x = patch.get_x() + patch.get_width() /2 
#막대 중심 x 좌표
    y = patch.get_height()
# patch : 각 막대
    axes.text(x,y,s = '%.2f' %(y),
              ha = 'center')
    
    
    
    
    
    
    
#longform to wideform


diff_melt.pivot_table(index = 'log_month',
                      columns = 'group',
                      values = 'count')










#중간고사 이후
#220428

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 23:59:22 2022

@author: jmjwj
"""

data
data.payment.replace({np.nan : 0}, inplace =True)
data['log_date'] = pd.to_datetime(data['log_date'])
data['month'] = data.log_date.dt.month

pay_count = data.groupby(['user_id', 'month']).payment.apply(lambda x : np.sum(x  > 0)).reset_index()
pay_mean = pay_count.groupby('month').payment.mean()
pay5 = pay_count.loc[pay_count.month == 5 ,'payment']
weights = np.ones_like(pay5) / len(pay5)
hist = pay5.plot.hist(bins = 20, weights = weights)
from scipy.stats import poisson
poi_x = np.arange(0, pay5.max())
poi_y = poisson.pmf(poi_x, mu = pay_mean.loc[5])
hist.plot(poi_x + 0.5 , poi_y)

import statsmodels.api as sm
zeroinf = sm.ZeroInflatedPoisson(pay5.values,
                                 np.ones([pay5.shape[0],1]),
                                 exog_infl = np.ones([pay5.shape[0], 1]))
#회귀분석의 한 방법. poisson 상황에서 0을 가지는 case가 너무 많을때 사용한다.
zeroinf.fit().summary()
def zeroinf_poi(x, mu, prob):
    y = prob * (x==0) + \
        (1-prob) * poisson.pmf(x, mu = mu)
    return y
hist = pay5.plot.hist(bins = 20, weights = weights)
poi_x = np.arange(0, pay5.max())
poi_y = list(
    map(lambda x : zeroinf_poi(x= x,
                               mu = np.exp(zeroinf.fit().params[1]),
                               prob = 1/ (1 + np.exp(-zeroinf.fit().params[0]))),
        poi_x))
hist.plot(poi_x + 0.5, poi_y)

zeroinf.fit().params

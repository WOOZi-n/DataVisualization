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







































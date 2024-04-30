#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install pulp')


# In[86]:


import pandas as pd
import pulp
from math import radians, cos, sin, asin, sqrt
import os
from sklearn.cluster import KMeans
import requests, json
import requests, json, pprint


# In[98]:


#df: 시설 정보 데이터프레임


# In[2]:


# Clean up and setup DataFrame
df.columns = ['name', 'lat', 'long', 'facility']  # Renaming for clarity
df['coordinates'] = list(zip(df['lat'], df['long']))  # Combine lat and long into a single tuple of coordinates

# Define a function to calculate the Haversine distance between two points
def haversine(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371  # Earth radius in kilometers

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    return R * c

total_origins = len(df['name'])
count = 0

# Create a distance matrix
dist_df = pd.DataFrame(index=df['name'], columns=df['name'])
for origin in df['name']:
    for destination in df['name']:
        if origin == destination:
            dist_df.at[origin, destination] = 0
        else:
            dist = haversine(df.loc[df['name'] == origin, 'coordinates'].values[0], 
                             df.loc[df['name'] == destination, 'coordinates'].values[0])
            dist_df.at[origin, destination] = dist
        
    count += 1
    print(f"Completed {count} out of {total_origins} origins")
    


# In[50]:


df_long = pd.melt(df, var_name='temp', value_name='distance')


# In[52]:


df_long.drop('temp', axis=1, inplace=True) 


# In[17]:


lat = df_or['x']
lng = df_or['y']
xy=lat.astype(str) + ", " +lng.astype(str)


# In[46]:


from itertools import product

# 모든 가능한 쌍의 조합을 생성
combinations = list(product(xy, repeat=2))

# 데이터 프레임 생성
final= pd.DataFrame(combinations, columns=['origin', 'destination'])


# In[77]:


final['identifier'] = final.apply(lambda x: '_'.join(sorted([x['origin'], x['destination']])), axis=1)

# identifier를 기준으로 중복 제거 (첫 번째 항목 유지)
final = final.drop_duplicates(subset='identifier', keep='first')

# 거리가 0인 행 제거
final = final[final['distance'] != 0]

# identifier 컬럼 제거 (더 이상 필요 없음)
final = final.drop(columns=['identifier'])


# In[66]:


def solve_p_median(df, p):
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
    
    locations = df['origin'].unique()
    destinations = df['destination'].unique()

    # 문제 정의
    prob = pulp.LpProblem('p_median', pulp.LpMinimize)

    # 변수
    x = pulp.LpVariable.dicts('x', [(o, d) for o in locations for d in destinations if (o, d) in zip(df['origin'], df['destination'])], 0, 1, pulp.LpBinary)
    y = pulp.LpVariable.dicts('y', {loc: pulp.LpVariable(f'y_{loc}', 0, 1, pulp.LpBinary) for loc in locations})

    # 목적 함수
    prob += pulp.lpSum(df.iloc[i]['distance'] * x[df.iloc[i]['origin'], df.iloc[i]['destination']] for i in range(len(df)))

    # 제약 조건
    for origin in locations:
        if origin in x:  # 이 조건을 추가하여 x 딕셔너리에 origin 키가 존재하는지 확인
            prob += pulp.lpSum(x[origin][destination] for destination in destinations if destination in x[origin]) == 1
            for destination in destinations:
                if destination in x[origin]:  # 이 조건을 추가하여 x[origin] 딕셔너리에 destination 키가 존재하는지 확인
                    prob += x[origin][destination] <= y[destination]

    prob += pulp.lpSum(y[location] for location in locations) == p

    # 문제 풀기
    prob.solve()

    # 선택된 위치 반환
    medians = [location for location in locations if pulp.value(y[location]) == 1]
    return medians


# In[94]:


solve_p_median(df, 14)


# In[107]:


cl1 = pd.read_csv(r"C:\Users\user\Desktop\2차 프젝\cluster0.csv")
cl2 = pd.read_csv(r"C:\Users\user\Desktop\2차 프젝\cluster1.csv")
cl3 = pd.read_csv(r"C:\Users\user\Desktop\2차 프젝\cluster2.csv")
cl4 = pd.read_csv(r"C:\Users\user\Desktop\2차 프젝\cluster3.csv")
cl5 = pd.read_csv(r"C:\Users\user\Desktop\2차 프젝\cluster4.csv")


# In[111]:


cl2.drop(["Unnamed: 0"], axis=1, inplace=True)
cl3.drop(["Unnamed: 0"], axis=1, inplace=True)
cl4.drop(["Unnamed: 0"], axis=1, inplace=True)
cl5.drop(["Unnamed: 0"], axis=1, inplace=True)
cl1.drop(["Unnamed: 0"], axis=1, inplace=True)


# In[115]:


result['Last_Word'] = result['Location'].apply(lambda x: x.split()[-1])

# 각 Last_Word가 cl1의 loc 열에 몇 번 나타나는지 계산
result['Most_Common_cl'] = result['Last_Word'].apply(lambda last_word: cl1['loc'].str.contains(last_word).sum())
result['Most_Common_c2'] = result['Last_Word'].apply(lambda last_word: cl2['loc'].str.contains(last_word).sum())
result['Most_Common_c3'] = result['Last_Word'].apply(lambda last_word: cl3['loc'].str.contains(last_word).sum())
result['Most_Common_c4'] = result['Last_Word'].apply(lambda last_word: cl4['loc'].str.contains(last_word).sum())
result['Most_Common_c5'] = result['Last_Word'].apply(lambda last_word: cl5['loc'].str.contains(last_word).sum())


# In[117]:


def find_max_column_name(row):
    # Most_Common_cl부터 Most_Common_c5까지의 열만 선택
    columns = ['Most_Common_cl', 'Most_Common_c2', 'Most_Common_c3', 'Most_Common_c4', 'Most_Common_c5']
    # 해당 행에서 가장 큰 값을 가진 열의 이름을 찾음
    max_column_name = columns[row[columns].astype(int).argmax()]
    return max_column_name

# 새로운 열 추가
result['cluster'] = result.apply(find_max_column_name, axis=1)


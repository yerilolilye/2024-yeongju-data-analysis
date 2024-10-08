#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from geopy.geocoders import Nominatim
import time
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import requests, json, pprint


#df는 시설들 설명과 위치 정보가 포함된 dataframe
#전처리: 필요없는 열 제거 및 좌표 -> 행정동 변환


df = df[['x', 'y', '시설명']]
lat = df['x']
lng = df['y']
df.drop('Unnamed: 0', axis=1, inplace=True)

# In[63]:


xy=lat.astype(str) + ", " +lng.astype(str) 


# In[99]:


def get_address(lat, lng):
    url = "https://dapi.kakao.com/v2/local/geo/coord2regioncode.json?x="+lng+"&y="+lat
  
    headers = {"Authorization": "KakaoAK kakao api key"}
    api_json = requests.get(url, headers=headers)
    full_address = json.loads(api_json.text)
    first_address_name = full_address['documents'][0]['address_name'] if full_address['documents'] else None
    return first_address_name



# In[106]:


location = []
for coord in xy:
    lat, lng = coord.split(', ')  # 각 좌표를 콤마로 분리하고 공백 제거
    address = get_address(lat.strip(), lng.strip())
    print(address)
    location.append(address)


# In[108]:


loc = []

# 리스트 순회 및 처리
for a in location:
    last_word = a.split()[-1]  # 주소를 공백 기준으로 분리하고 마지막 단어 선택
    loc.append(last_word)

# 결과 출력
print(len(loc))


#비정형 데이터 변형 

# In[118]:


facility_types = df['시설명'].unique()


# In[119]:


facility_to_index = {name: index for index, name in enumerate(facility_types)}

# '시설명' 열의 각 값에 대한 숫자 매핑을 '시설명1' 열로 추가
df['시설명1'] = df['시설명'].map(facility_to_index)

# 결과 출력
print(df)


# In[ ]:


#클러스터링 군집 개수 최적화


# In[38]:


sse = {}
silhouette_coefficients = []
for k in range(2, 11):  # 1개의 클러스터는 실루엣 계수를 계산할 수 없으므로 2부터 시작
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)
    kmeans.fit(X)
    sse[k] = kmeans.inertia_  # SSE 값 저장
    score = silhouette_score(X, kmeans.labels_)
    silhouette_coefficients.append(score)

# 엘보우 그래프 그리기
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(list(sse.keys()), list(sse.values()), 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.title('Elbow Method For Optimal k')

# 실루엣 계수 그래프 그리기
plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_coefficients, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient')
plt.title('Silhouette Score For Each k')
plt.show()


# In[47]:


#클러스터링 모델링


# In[132]:


X = df[['x', 'y', '시설명1']]

# KMeans 모델 생성 및 학습
kmeans = KMeans(n_clusters=5, ra화

plt.figure(figsize=(10, 6))
colors = plt.cm.Blues(np.linspace(0.3, 0.9, kmeans.n_clusters))  # 동적으로 색상 생성

for i in range(kmeans.n_clusters):
    cluster_data = df[df['cluster'] == i]
    plt.scatter(cluster_data['x'], cluster_data['y'], color=colors[i], label=f'Cluster {i}', s=100, edgecolor='black')


plt.title('K-Means Clustering of Facilities')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.legend()
plt.grid(True)
plt.show()


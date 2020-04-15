import numpy as np
import pandas as pd
from pandas import plotting
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

#data imports and some cleaning 

data = pd.read_csv('Mall_Customers.csv')
data.head(10)
data.reset_index()
data.tail(10)
data.isna().any(axis =1)

data_cleaned= data.drop("CustomerID",axis =1)

#Visualization

sns.set(style='whitegrid')
sns.pairplot(data=data)

from pandas import plotting
plt.rcParams['figure.figsize'] = (15, 10)
plotting.andrews_curves(data.drop("CustomerID", axis=1), "Gender")
plt.title('Andrew Curves for Gender', fontsize = 20)
plt.show()

sns.set(style='whitegrid')
sns.pairplot(data_cleaned)

plt.subplot(1,2,1)
sns.set(style='whitegrid')
sns.distplot(data['Annual Income (k$)'])
plt.title('Distribution of Annual Income',fontsize=20)
plt.xlabel('Range of Annual Income')
plt.ylabel('Count')

plt.subplot(1,2,2)
sns.set(style='whitegrid')
sns.distplot(data['Age'],color='red')
plt.title('Distribution of Age',fontsize=20)
plt.xlabel('Range of Age')
plt.ylabel('Count')

data_new=data_cleaned.dropna(axis=0)
sns.set(style='whitegrid')
sns.distplot(data['Spending Score (1-100)'],color='green')
plt.title('Distribution of Spending Score',fontsize=20)
plt.xlabel('Range of Spendinf Score')
plt.ylabel('Count')

#Calculating correlation visulaization

coore= data_cleaned.corr
print(coore)

sns.stripplot(x='Annual Income (k$)',y='Gender',data=data_cleaned)
sns.set(style='whitegrid')
sns.stripplot(x='Age',y='Spending Score (1-100)',data=data_cleaned)


#Piechart to check the %cntage of Males/Females in the customers list

labels = ['Female', 'Male']
size = data['Gender'].value_counts()
colors = ['lightgreen', 'orange']
explode = [0, 0.1]

plt.pie(size,
    explode=explode,
    labels=labels,
    colors=colors,
    autopct='%.2f%%',
    pctdistance=0.6,
    shadow=False,
    labeldistance=1.1,
    startangle=None,
    radius=None,
    counterclock=True,
    wedgeprops=None,
    textprops=None,
    center=(0, 0),
    frame=False,
    rotatelabels=False,
    data=data_cleaned)

plt.title('Gender', fontsize = 20)
plt.axis('off')
plt.legend()
plt.show()

plt.figure(figsize=(15,8))
sns.heatmap(data.corr(),annot=True,cmap='Wistia')
plt.title('Heatmap for the Data', fontsize = 20)
plt.show()


x= data_cleaned['Age']
y=data_cleaned['Annual Income (k$)']
z=data_cleaned['Spending Score (1-100)']

sns.lineplot(x,y, color='blue')
sns.lineplot(x,z,color='red')

plt.title('Age vs Annual Income and Spending Score')

sns.stripplot(x='Gender',y='Age',hue ='Gender',data=data_cleaned)

sns.violinplot(x='Gender',y='Annual Income (k$)',data =data_cleaned,palette ='rainbow')
sns.boxplot(x='Gender',y='Annual Income (k$)',palette='rainbow',data=data_cleaned)

sns.boxplot(x='Gender',y='Spending Score (1-100)',palette='viridis',data=data_cleaned)

data_cleaned.pivot_table('Spending Score (1-100)',columns ='Age', index='Gender')

data_cleaned.head(2)

#Standard Scalar 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler = StandardScaler()
data_df= scaler.fit_transform(data_cleaned[['Age','Annual Income (k$)','Spending Score (1-100)']])

#Creating Dendogram

import scipy.cluster.hierarchy as sch
plt.figure(figsize=(15,8))
x = data.iloc[:, [3, 4]].values
dendrogram = sch.dendrogram(sch.linkage(data_df, method = 'ward'))
plt.title('Dendrogam', fontsize = 20)
plt.xlabel('Customers')
plt.ylabel('Ecuclidean Distance')
plt.show()

#Creating Model KMeans

from sklearn.cluster import KMeans
wcss =[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

#Elbow Method Curve

plt.rcParams['figure.figsize']=(15,5)
plt.plot(range(1,11),wcss)
plt.title('K-Means(Elbow Method)',fontsize =20)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Count')
plt.grid()

clusters.cluster_centers_
from sklearn.cluster import KMeans
k=3
clusters=KMeans(k,random_state=42)
clusters.fit(data_df)
data["cluster_id"]=clusters.labels_

data.groupby('cluster_id')['Spending Score (1-100)','Annual Income (k$)','Age'].agg(["mean","std"]).reset_index()
plt.style.use('fivethirtyeight')

plt.figure(figsize=(15,8))
sns.scatterplot("Spending Score (1-100)","Age",data =data, hue ="cluster_id")

#NOw checking which datasets belong to which clusters

data[data.cluster_id==0]
data[data.cluster_id==1]
data[data.cluster_id==2]

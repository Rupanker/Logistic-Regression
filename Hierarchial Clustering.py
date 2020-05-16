#Hierarchial Clustering
#import the libraries
import numpy  as np
import matplotlib.pyplot as plt
import pandas as pd 

#import the dataset
dataset=pd.read_csv("Mall_Customers.csv")
x= dataset.iloc[:, 3:5].values

#Using the Dendogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendogram=sch.dendrogram(sch.linkage(x,method='ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

#Fitting Hierarchical Clustering to the mall dataset
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(x)


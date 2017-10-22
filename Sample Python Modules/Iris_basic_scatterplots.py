#from sklearn.metrics import accurracy
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame 


# Loading and Viewing Iris Data set from top level
iris_data=datasets.load_iris()
print iris_data.keys()
print "---------",iris_data["DESCR"]
print "-----'feature_names'-----\n",iris_data["feature_names"]
print "------'data'-----------\n",iris_data["data"]
print "------'target'-----------\n",iris_data["target"]
print "------'target_names'-----------\n",iris_data["target_names"]


#Loading data into Pandas
data=DataFrame(data=iris_data["data"], columns=iris_data["feature_names"])
#using 2 diff ways of accessing dictionary keys
# Assign column our own name
target=DataFrame(data=iris_data.target,columns=["Species"])

#No normalization is needed
print np.shape(iris_data["target"]),np.shape(iris_data["data"])
print np.shape(data),np.shape(target)

#Plotting our iris_data with --- Scatter plots
fig,axes=plt.subplots(nrows=2,ncols=2,figsize=plt.figaspect(5./20))
axes[0,0].set(title="Sepal len v/s Length")
axes[1,0].set(title="Sepal width  v/s Length")
axes[0,1].set(title="Petal len  v/s Length")
axes[1,1].set(title="Petal width  v/s Length")


axes[0,1].scatter(data["sepal length (cm)"],target["Species"])
axes[1,1].scatter(data["sepal width (cm)"], target["Species"])
axes[0,0].scatter(data["petal length (cm)"],target["Species"])
axes[1,0].scatter(data["petal width (cm)"], target["Species"])
for x,y in [(x,y) for x in range(2) for y in range(2)]:
    axes[x,y].legend()
    axes[x,y].grid(True)
plt.suptitle("IRIS DATA SET", fontsize='20')
plt.show()


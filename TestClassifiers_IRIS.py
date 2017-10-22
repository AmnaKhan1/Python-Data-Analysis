#from sklearn.metrics import accurracy
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
# Getting away with the import of scikit learn Common models
#------------ Linear Clssifier -------------------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
#---------- Non-Linear Classifier -------------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB #Linear in certain criteria
#---------------------------------------------------
from sklearn.model_selection import train_test_split



#-----------------------------------------------------------
# Loading and Viewing Iris Data set from top level
#-----------------------------------------------------------
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

#-------------------------------------------------------
#Plotting our iris_data with --- *****Scatter plots******
#--------------------------------------------------------
fig,axes=plt.subplots(figsize=(10,10))

#Extracting features
x=np.array(data["sepal length (cm)"])
y=np.array(data["sepal width (cm)"])
#Extracting unique classes
unique_Classes=list(set(target["Species"]))
#------------------------------------------------------------------
#considering only 1st two features of Sepal:
#Print species relationship with sepal length and sepal width
#-------------------------------------------------------------------
for counter,u in enumerate(unique_Classes):
    print i,u
    xcounter=[x[j] for j in range(x.size) if target["Species"][j]==u]
    ycounter=[y[j] for j in range(y.size) if target["Species"][j]== u]
    plt.scatter(xcounter,ycounter,label=str(u))
plt.legend()
plt.grid(True)
plt.show()

#---------------------------  Conclusion   ---------------------
# The graph indiactes that the 1st two features can best 
# separate Setosa from the other two
#---------------------------------------------------------------

#--------------------------------------------------------
#************* Testing Models*************************
#--------------------------------------------------------
print "Splitting data"
X=np.array(data["sepal length (cm)"],data["sepal width (cm)"])
Y=np.array(target["Species"])
X_train,X_test,y_train,y_test=train_test_split(data,target,
                                               test_size=0.33,random_state=42)
test=np.ravel(y_train)
print "data Splitted\n"
print "X_train.shape= ",X_train.shape,"\ny_train.shape=",y_train.shape\
            ,"X_test.shape=",X_test.shape,"y_test=",y_test.shape,"\n"\
            ,"After ravel on y_train",test.shape,"\n"

#Making the list of classifers
penalty=0
classifiers=[('KNN',KNeighborsClassifier()),
             ('LDA',LinearDiscriminantAnalysis()),
#             ('linearRegression',LinearRegression()),
             ('Logistic regression',LogisticRegression()),
             ('SVM',SVC()),
             ('DT',DecisionTreeClassifier()),
             ('NB',GaussianNB())]

#setting random penalty range
Lambda_inverse=10**np.arange(0,12,0.25)
accuracies=[]
#Ready to run classifiers
for names,clf in enumerate(classifiers):
        print "applying classifier: ",clf[0]
        clf[1].fit(X_train,np.ravel(y_train))
        y_hat=clf[1].predict(X_test)
        accuracy=100.0*np.mean(y_hat ==np.ravel(y_test))
        print "*accuracy",accuracy


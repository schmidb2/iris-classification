import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tabulate import tabulate 


data = pd.read_csv('iris_data.csv')

data_y = data['species']
data_x = data.drop(columns = ['species'])

x_train,x_test,y_train,y_test = train_test_split(data_x,data_y,test_size=0.4,stratify = data['species'], random_state = 42)

results_table = [['Model','Accuracy Score']]

#Decision Tree

mod_dt = DecisionTreeClassifier(max_depth=3,random_state=1)
mod_dt.fit(x_train,y_train)
predictions_dt = mod_dt.predict(x_test)
accuracy_dt = accuracy_score(y_test,predictions_dt)
results_table.append(['Decision Tree',accuracy_dt])

#Gaussian Naive Bayes

mod_gnb = GaussianNB()
mod_gnb.fit(x_train,y_train)
predictions_gnb = mod_gnb.predict(x_test)
accuracy_gnb = accuracy_score(y_test,predictions_gnb)
results_table.append(['Gaussian Naive Bayes',accuracy_gnb])


#Linear Discriminant Analysis

mod_lda = LinearDiscriminantAnalysis()
mod_lda.fit(x_train,y_train)
predictions_lda = mod_lda.predict(x_test)
accuracy_lda = accuracy_score(y_test,predictions_lda)
results_table.append(['Linear Discriminant Analysis',accuracy_lda])

#Quadratics Discriminant Analysis

mod_qda = QuadraticDiscriminantAnalysis()
mod_qda.fit(x_train,y_train)
predictions_qda = mod_qda.predict(x_test)
accuracy_qda = accuracy_score(y_test,predictions_qda)
results_table.append(['Quadratic Discriminant Analysis',accuracy_qda])

#K Nearest Neighbour

accuracy_knn = np.empty(10)
for i in range(1,11):
    mod_knn = KNeighborsClassifier(n_neighbors=i)
    mod_knn.fit(x_train,y_train)
    predictions_knn = mod_knn.predict(x_test)
    accuracy_knn[i-1] = accuracy_score(y_test,predictions_knn)
    results_table.append(['K Nearest Neightbour (k='+str(i)+')',accuracy_knn[i-1]])

#print results
print(tabulate(results_table,headers='firstrow',tablefmt='fancy_grid'))

#create confusion matrix
#plot best performing models
    
matrix_dt = confusion_matrix(y_test,predictions_dt)
matrix_lda = confusion_matrix(y_test,predictions_lda)
matrix_qda = confusion_matrix(y_test,predictions_qda)

fig,axes = plt.subplots(1,3,figsize=(20,4))
fig.suptitle('Confusion Matrices')

p1 = sn.heatmap(matrix_dt, ax=axes[0],annot=True,fmt='d')
axes[0].set_title('Decision Tree')
sn.heatmap(matrix_lda, ax=axes[1],annot=True,fmt='d')
axes[1].set_title('Linear Discriminant Analysis')
sn.heatmap(matrix_qda, ax=axes[2],annot=True,fmt='d')
axes[2].set_title('Quadratic Discriminant Analysis')
plt.show()














    

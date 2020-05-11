
# We are trying to predict whether a patient has AD.
from sklearn.model_selection import train_test_split
from os.path import expanduser, join
import numpy as np
import pandas as pd # 引用套件並縮寫為 pd

# fADNI = join('L:/AI_Related/','NC_AD.xlsx')
fADNI = ('NC_AD.xlsx')
print(fADNI)

fADNI_xlsx_df = pd.read_excel(fADNI)
fADNI_xlsx_df.head()# 觀察前五個觀測值
# print(fADNI_xlsx_df.shape)
# print(fADNI_xlsx_df.head())
# print(fADNI_xlsx_df[0:5])
# print(fADNI_xlsx_df.Group)

X=fADNI_xlsx_df[['APOE_Status', 'Age','Sex','Edu','TIV','CH123_L','CH123_R','CH4_L','CH4_R','Amy_L','Amy_R','MEM','EF']] # Features
y=fADNI_xlsx_df['Group']  # Labels

##### Split dataset into training set and test set #####
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) # 70% training and 30% test
print(X_test.shape)
print(X_train.shape)


#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
#在scikit-learn中，RF的分类类是RandomForestClassifier，回归类是RandomForestRegressor。
clf=RandomForestClassifier(n_estimators=100, random_state=0) #森林中樹木的數量(n_estimators, base estimator的數量)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
# print(y_pred)
# print(y_test)

# print(y_pred.shape)
# print(y_test.shape)

# print(X_test.shape)
# print(X_test)

########################################################

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt

# print(fADNI_xlsx_df.iloc[:,10]) ## column of Amy_L
# print(fADNI_xlsx_df.iloc[:,12]) ## column of MEM

#plt.scatter(fADNI_xlsx_df.iloc[:,10],fADNI_xlsx_df.iloc[:,12])
#plt.scatter(fADNI_xlsx_df.iloc[1:168,10],fADNI_xlsx_df.iloc[1:168,12], cmap=ListedColormap(['r', 'y', 'b']),edgecolor='k', s=20)

from matplotlib.colors import ListedColormap, LinearSegmentedColormap 

# plt.scatter(fADNI_xlsx_df.iloc[:,12],fADNI_xlsx_df.iloc[:,10], c=y,cmap=ListedColormap(['r', 'y', 'b']),edgecolor='k', s=10) 
# plt.xlabel('MEM_00')
# plt.ylabel('Amy_L_00')
# plt.legend('')
# plt.show()

# plt.scatter(fADNI_xlsx_df.iloc[:,12],fADNI_xlsx_df.iloc[:,10], c=y,cmap=ListedColormap(['r', 'y', 'b']),s=10) 
# plt.scatter(X_test.iloc[:,12], X_test.iloc[:,10], c=y_test,cmap=ListedColormap(['r', 'y', 'b']),edgecolor='g',s=50)# Plot th prediction results from test set
# plt.xlabel('MEM_01')
# plt.ylabel('Amy_L_01')
# plt.legend('')
# plt.show()

# plt.scatter(fADNI_xlsx_df.iloc[:,12],fADNI_xlsx_df.iloc[:,10], c=y,cmap=ListedColormap(['r', 'y', 'b']),s=10) 
# plt.scatter(X_test.iloc[:,12], X_test.iloc[:,10], c=y_pred,cmap=ListedColormap(['r', 'y', 'b']),edgecolor='m',s=150)# Plot th prediction results from test set
# plt.scatter(X_test.iloc[:,12], X_test.iloc[:,10], c=y_test,cmap=ListedColormap(['r', 'y', 'b']),edgecolor='g',s=50)# Plot th prediction results from test set
# plt.xlabel('MEM_02')
# plt.ylabel('Amy_L_02')
# plt.legend('')
# plt.show()

# Circle out the incorrect predictions
# loc > localized
X_wrong = X_test.loc[y_pred != y_test]
# print(X_wrong)


# c= y>> use y(group) as the factor divide two value 
plt.scatter(fADNI_xlsx_df.iloc[:,12],fADNI_xlsx_df.iloc[:,10], c=y,cmap=ListedColormap(['r', 'y', 'b']),s=10) 
# X_train >87 
plt.scatter(X_test.iloc[:,12], X_test.iloc[:,10], c=y_test,cmap=ListedColormap(['r', 'y', 'b']),edgecolor='g',s=50)# Plot th prediction results from test set
# error pridiction >5 
plt.scatter(X_wrong.iloc[:,12], X_wrong.iloc[:,10], s=150, edgecolor='k', facecolors='k', zorder=50)

plt.legend('Original TEST Wrong')
plt.xlabel('MEM')
plt.ylabel('Amy_L')
plt.show()

#plt.scatter(X_wrong.iloc[:,12], X_wrong.iloc[:,10], s=150, facecolors='g', zorder=10)
#plt.scatter(fADNI_xlsx_df.iloc[:,12],fADNI_xlsx_df.iloc[:,10], c=y,cmap=ListedColormap(['r', 'y', 'b']),s=10) 
#plt.scatter(X_test.iloc[:,12], X_test.iloc[:,10], c=y_test,cmap=ListedColormap(['r', 'y', 'b']),edgecolor='g',s=50)# Plot th prediction results from test set






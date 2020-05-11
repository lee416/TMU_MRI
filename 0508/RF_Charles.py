
from sklearn.model_selection import train_test_split
from os.path import expanduser, join
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import ListedColormap, LinearSegmentedColormap 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

##### read data #####################################
TCIA = ('TCIA_DHA_GBM.xlsx')
TCIA_pd = pd.read_excel(TCIA)
# print(TCIA_pd.columns.ravel())
X_header = TCIA_pd.columns.ravel()
X_header = np.delete(X_header,0)

y = TCIA_pd['overall_survival(1yr boundary)']
X = TCIA_pd[X_header]
#####################################################



##### Split dataset into training set and test set #####
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) # 70% training and 30% test
# print(X_test.shape)
# print(X_train.shape)

#Create a Gaussian Classifier
#在scikit-learn中，RF的分类类是RandomForestClassifier，回归类是RandomForestRegressor。
clf = RandomForestClassifier(n_estimators=100, random_state=0) #森林中樹木的數量(n_estimators, base estimator的數量)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


X_wrong = X_test.loc[y_pred != y_test]


plt.scatter(fADNI_xlsx_df.iloc[:,12],fADNI_xlsx_df.iloc[:,10], c=y,cmap=ListedColormap(['r', 'y', 'b']),s=10) 
plt.scatter(X_test.iloc[:,12], X_test.iloc[:,10], c=y_test,cmap=ListedColormap(['r', 'y', 'b']),edgecolor='g',s=50)# Plot th prediction results from test set
plt.scatter(X_wrong.iloc[:,12], X_wrong.iloc[:,10], s=150, edgecolor='k', facecolors='k', zorder=50)


plt.legend('TCIA')
plt.xlabel('MEM')
plt.ylabel('Amy_L')
plt.show()




# print(fADNI_xlsx_df.iloc[:,10]) ## column of Amy_L
# print(fADNI_xlsx_df.iloc[:,12]) ## column of MEM

#plt.scatter(fADNI_xlsx_df.iloc[:,10],fADNI_xlsx_df.iloc[:,12])
#plt.scatter(fADNI_xlsx_df.iloc[1:168,10],fADNI_xlsx_df.iloc[1:168,12], cmap=ListedColormap(['r', 'y', 'b']),edgecolor='k', s=20)


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






# -- coding: utf-8 --
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score

"""VERİLER YÜKLEDİM, İNCELEDİM, GÖRSELLEŞTİRDİM..."""
veriler = pd.read_csv('veri.csv')
veriler.isna().sum()
veriler.describe()
veriler.info()

#ID KOLONUNU SİLİYORUM
veriler.drop('id',inplace  =True,axis = 1)

#GÖRSELLEŞTİRİYORUM
plt.figure(figsize=(4,4))
sns.countplot(data = veriler,x = 'diagnosis')

sns.pairplot(veriler, hue="diagnosis", vars=["radius_mean", "texture_mean", "perimeter_mean", "radius_worst", "perimeter_worst"])
plt.show()

#DIAGNOSIS KOLONUMU SAYISALLAŞTIRIYORUM M/B --> 1/0
diagnosis = veriler[["diagnosis"]]
diagnosis = preprocessing.LabelEncoder().fit_transform(diagnosis)

#DIAGNOSIS KOLONUNU TAHMİN ETTİRECEĞİM İÇİN KALAN VERİLERİ AYIRIYORUM
kalan = veriler.iloc[:,2:].values

#DIAGNOSIS KOLONUNU SAYILAŞTIRDIKTAN SONRA TEKRAR KALAN VERİLERLE BİRLEŞTİRİYORUM
bir= pd.DataFrame(data= diagnosis, index= range(569), columns= ["diagnosis"])
iki= pd.DataFrame(data= kalan, index= range(569))
veri=pd.concat([bir,iki], axis=1)

"""EĞİTİM/TEST VERİLERİ AYRILIYOR..."""
x_train, x_test,y_train,y_test = train_test_split(kalan,diagnosis,test_size=0.33, random_state=0)

"""ÖLÇEKLEME YAPIYORUM..."""
sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

print("********** ALGORİTMALAR DENENİYOR ***************")  

#LOGİSTİC REGRESSİON
from sklearn.linear_model import LogisticRegression
print('LGR')
logr = LogisticRegression(random_state=0)
logr.fit(x_train,y_train)

y_pred = logr.predict(x_test)
acc= accuracy_score(y_test, y_pred)
print("DOĞRULUK: ", acc)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True, fmt='d')
print("\nLGR",classification_report(y_test, y_pred))
print("Roc eğrisi altındaki alan (AUC): ", roc_auc_score(y_test, y_pred))


print("*******")
#KNN
from sklearn.neighbors import KNeighborsClassifier
print('KNN')
knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)
acc= accuracy_score(y_test, y_pred)
print("DOĞRULUK: ", acc)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True, fmt='d')
print("\nKNN",classification_report(y_test, y_pred))
print("Roc eğrisi altındaki alan (AUC): ", roc_auc_score(y_test, y_pred))


print("*******")
#SVC
from sklearn.svm import SVC
print('SVC')
svc = SVC(kernel='rbf')
svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)
acc= accuracy_score(y_test, y_pred)
print("DOĞRULUK: ", acc)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True, fmt='d')
print("\nSVC",classification_report(y_test, y_pred))
print("Roc eğrisi altındaki alan (AUC): ", roc_auc_score(y_test, y_pred))


print("*******")
#NAİVEBAYES
from sklearn.naive_bayes import GaussianNB
print('GNB')
gnb = GaussianNB()  
gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test)
acc= accuracy_score(y_pred,y_test)
print("DOĞRULUK: ", acc)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True, fmt='d')
print("\nGNB",classification_report(y_test, y_pred))
print("Roc eğrisi altındaki alan (AUC): ", roc_auc_score(y_test, y_pred))


print("*******")
#KARARAGACI
from sklearn.tree import DecisionTreeClassifier
print('DTC')
dtc = DecisionTreeClassifier(criterion = 'entropy')
dtc.fit(x_train,y_train)

y_pred = dtc.predict(x_test)
acc= accuracy_score(y_test, y_pred)
print("DOĞRULUK: ", acc)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True, fmt='d')
print("\nDTC",classification_report(y_test, y_pred))
print("Roc eğrisi altındaki alan (AUC): ", roc_auc_score(y_test, y_pred))


print("*******")
#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
print('RFC')
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy', random_state=500)
rfc.fit(x_train,y_train)

y_pred = rfc.predict(x_test)
acc= accuracy_score(y_pred,y_test)
print("DOĞRULUK: ", acc)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True, fmt='d')
print("\nRFC",classification_report(y_test, y_pred))
print("Roc eğrisi altındaki alan (AUC): ", roc_auc_score(y_test, y_pred))


print("****** CROSS VALUDATION *********")
#CROSS VALIDATION
from sklearn.model_selection import cross_val_score
k=10

crossval =cross_val_score(logr, X= x_train, y= y_train, cv=k)
print("LOGR Cross Validation: ", np.mean(crossval))
print("Cross Validation Standart Sapma: ",np.std(crossval))


crossval =cross_val_score(svc, X= x_train, y= y_train, cv=k)
print("SVC Cross Validation: ", np.mean(crossval))
print("Cross Validation Standart Sapma: ",np.std(crossval))


crossval =cross_val_score(rfc, X= x_train, y= y_train, cv=k)
print("RFC Cross Validation: ", np.mean(crossval))
print("Cross Validation Standart Sapma: ",np.std(crossval))


crossval =cross_val_score(dtc, X= x_train, y= y_train, cv=k)
print("DTC Cross Validation: ", np.mean(crossval)) 
print("Cross Validation Standart Sapma: ",np.std(crossval))


crossval =cross_val_score(gnb, X= x_train, y= y_train, cv=k)
print("GNB Cross Validation: ", np.mean(crossval))
print("Cross Validation Standart Sapma: ",np.std(crossval))


crossval =cross_val_score(knn, X= x_train, y= y_train, cv=k)
print("KNN Cross Validation: ", np.mean(crossval))
print("Cross Validation Standart Sapma: ",np.std(crossval))
%matplotlib inline
#Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns

#Data processing Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#Machine model Algorithm module
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_squared_error

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score

from sklearn.metrics import RocCurveDisplay

from sklearn.metrics import auc

from sklearn.metrics import roc_curve

from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.metrics import log_loss

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from IPython.display import Image

import graphviz

from sklearn.tree import export_graphviz

df = pd.read_excel("Rice_Osmancik_Cammeo_Dataset.xlsx")

df

df.head()

df.tail()

df['CLASS'].unique() #Two Rice Types 'Cammeo', 'Osmancik'

for i in range(len(df['CLASS'])):
    if(df.iloc[i,7]!='Cammeo'):
        print(df.iloc[i,7],i)
        break 

df.info()

df.isnull()

True in df.isnull() #checks missing values
#Missing values are NONE

df.describe()

df.corr()

plt.figure(figsize=(13,7))
plt.title("Correlation Among Features")
sns.heatmap(df.corr(),cmap='coolwarm',linewidths=1,annot=True)
plt.show()

df.loc[df['CLASS']=='Cammeo'].describe()

df.loc[df['CLASS']=='Osmancik'].describe()

barx=[len([0 for i in df['CLASS'] if(i=='Cammeo')]),len([1 for i in df['CLASS'] if(i=='Osmancik')])]
bary=['Cammeo','Osmancik']
plt.figure(figsize=(20,10))
plt.suptitle('Dataset Visualisation',size='20')
plt.subplot(2, 4, 1)
plt.title('CLASS',color='r')
plt.bar(bary,barx,0.4,color=['r','tab:blue'])
plt.xlabel('Rice types')
plt.ylabel('No. of Rows')
plt.text(0,barx[0],barx[0])
plt.text(1,barx[1],barx[1])
plt.ylim([0,3000])
plt.subplot(2, 4, 2)
plt.title('AREA',color='r')
plt.plot(df['AREA'],color='Green')
plt.xlim([1400,2000])
plt.subplot(2, 4, 3)
plt.title('PERIMETER',color='r')
plt.xlim([1400,2000])
plt.plot(df['PERIMETER'],color='Orange')
plt.subplot(2, 4, 4)
plt.title('MAJORAXIS',color='r')
plt.xlim([1400,2000])
plt.plot(df['MAJORAXIS'],color='grey')
plt.subplot(2, 4, 5)
plt.title('MINORAXIS',color='r')
plt.xlim([1400,2000])
plt.plot(df['MINORAXIS'],color='k')
plt.subplot(2, 4, 6)
plt.title('ECCENTRICITY',color='r')
plt.xlim([1400,2000])
plt.plot(df['ECCENTRICITY'],color='blue')
plt.subplot(2, 4, 7)
plt.title('CONVEX_AREA',color='r')
plt.xlim([1400,2000])
plt.plot(df['CONVEX_AREA'],color='purple')
plt.subplot(2, 4, 8)
plt.title('EXTENT',color='r')
plt.xlim([1400,2000])
plt.plot(df['EXTENT'],color='slateblue')
plt.show()

for i in range(7):
    for j in range(i,7):
        if(i!=j):
            plt.figure(figsize=(7,4))
            plt.subplot(1,2,1)
            plt.suptitle(df.columns.values[i]+" VS "+df.columns.values[j])
            plt.title('Cammeo')
            plt.scatter(df.iloc[:1630,i],df.iloc[:1630,j],color='Red',marker='.')
            plt.subplot(1,2,2)
            plt.title('Osmancik')
            plt.scatter(df.iloc[1630:,i],df.iloc[1630:,j],color='tab:blue',marker='.')
    plt.show()

plt.figure(figsize=(5,5))  #No. of rows for each type of rice
plt.bar(bary,barx,0.4,color=['r','tab:blue'])
plt.title('CLASS',color='r')
plt.xlabel('Rice types',weight='bold')
plt.ylabel('No. of Rows',weight='bold')
plt.text(0,barx[0]+50,barx[0],size='x-large',ha='center',weight='bold')
plt.text(1,barx[1]+50,barx[1],size='x-large',ha='center',weight='bold')
plt.ylim([0,3000])
plt.show()

x=df.loc[:,df.columns!='CLASS']                       #all columns except CLASS column
y=df['CLASS']                                         #CLASS cloumn
y.head()                                              #X is features of Rice, Y is Rice types to predict

LE = LabelEncoder()
y= LE.fit_transform(y) #Cammeo=0, Osmancik=1
y  

def bestmodel(x_train,y_train,x_test,y_test):# Gives me Best Random forest model with optimal Hyper parameters 
    def ele(v):
        return v[2] 
    f2x=[]
    for i in range(1,11):
        for j in range(10,0,-1):
            model=RandomForestClassifier(n_estimators=100, bootstrap=True, max_depth=i, min_samples_leaf=j, random_state=12)
            rf_train=model.fit(x_train,y_train)
            f2=f1_score(y_test,rf_train.predict(x_test))
            lgls=log_loss(y_test, rf_train.predict_proba(x_test))
            acc=accuracy_score(y_test, rf_train.predict(x_test))
            rac=roc_auc_score(y_test, rf_train.predict(x_test))
            f2x.append([i,j,f2,lgls,rac,acc])
    f2x.sort(key=ele,reverse=True)   #sorts according to fscore in decending order
    for i in f2x:
        if(f2x[0][2]==i[2]):
            print(i)
        else:
            break

for i in np.arange(0.1,0.6,0.1):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=i, shuffle=True,random_state=12)
    print("Test_size= ",i)
    bestmodel(x_train,y_train,x_test,y_test)
#output format= [max_depth ,min_sample_leaf, fscore, log_loss,roc_auc_score,accuracy_score]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True,random_state=12)

model=RandomForestClassifier(n_estimators=100, bootstrap=True, max_depth=6, min_samples_leaf=10, random_state=12)
rf_train=model.fit(x_train,y_train)
rf_train
pred=model.predict(x_test)

len(pred)

confusion_matrix(y_test,rf_train.predict(x_test))

fig, ax = plt.subplots(figsize=(8, 8))
plt.rcParams.update({'font.size': 16})
cm=confusion_matrix(y_test, pred)
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
plt.savefig('confusion_matrix.jpg')
plt.show()

f1_score(y_test,rf_train.predict(x_test)) #F1_score balances the percision and recall from confusion matrix

precision_recall_fscore_support(y_test,rf_train.predict(x_test),average ='binary')
#output Format(precision, recall, fscore, support)

log_loss(y_test, rf_train.predict_proba(x_test))

accuracy_score(y_test, rf_train.predict(x_test))*100 #Accuracy of prediction in %

roc_auc_score(y_test, rf_train.predict(x_test))

fpr, tpr, thresholds = roc_curve(y_test, rf_train.predict(x_test))
roc_auc = auc(fpr, tpr)
display =RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,estimator_name='Random Forest')
display.plot(color='r')
plt.plot(list(np.arange(0,2,0.1)),list(np.arange(0,2,0.1)),color='k')
plt.title('ROC curve')
plt.xlim([0,1])
plt.ylim([0,1])
plt.show()

'''

Credit Card Fraud Detection dataset


'''
#Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler 
from sklearn.linear_model._perceptron import Perceptron
from sklearn.ensemble import RandomForestClassifier ,GradientBoostingClassifier
from sklearn.model_selection import train_test_split ,GridSearchCV
from sklearn.linear_model import LogisticRegression ,RidgeClassifier
from sklearn.metrics import confusion_matrix,f1_score ,roc_curve,roc_auc_score
from sklearn.svm import SVC
import joblib

#Load Data

df_=pd.read_csv(r'E:\Rev-DataScience\AI-ML\MLProjects_Structured_Data\creditcard.csv'
    ,encoding='latin')

#Info
print(df_.head(5))
print(df_.shape)
print(df_.describe())
print(df_.isnull().sum())
print(df_.dtypes)

#=========================================
#Optimization --
#=========================================
print(df_.memory_usage(deep=True))
"""
for col in df_.columns:
    if df_[col].dtype =='float64':
        df_[col]=df_[col].astype('float16')
"""
df_ = df_.astype({col: 'float32' for col in df_.select_dtypes('float64').columns})

df_['Class']=df_['Class'].astype('int8')

print(df_.memory_usage(deep=True))

#=========================================
#Outliers --
#=========================================


def find_outliers(series):
    Q1=series.quantile(0.25)
    Q3=series.quantile(0.75)
    IQR= Q3 - Q1
    lower= Q1 - 1.5 * IQR
    upper= Q3 + 1.5 * IQR

    return series[(series < lower) | (series > upper)]

'''
for col in df_.columns:
    print(f'Outliers in {col} => {find_outliers(df_[col]).shape[0]}')
    plt.boxplot(df_[col])
    plt.title(f" Outliers in {col} ")
    plt.show()

    # The Outliers in all columns --
     
'''     

def remove_outliers(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    return df_clean

num_cols = df_.select_dtypes(include=['float', 'int']).columns.tolist()
df_ = remove_outliers(df_, ['Amount', 'Time'])

#Removed 2 columns have sensitive outliers 

print("Before :", df_.shape)
print("After ", df_.shape)


'''
OverView Insights :-

-- Shape of dataframe => 
Rows = 284807 , columns = 31
-- Theres no empty values 
-- all data types = float64 except the target class column = int64
-- max of outliers row almost 30000
-- Alot of values zero class
   class zero = 284315 , class one = 492

** Action
-- optimize data type 
-- Removed sensitive outliers columns in amount , time 

'''
#=======================================
# Simple Relations 
#=======================================
print(df_['Class'].value_counts())


cols=[]

for col in df_.columns:
    if col=='Class':
        continue
    else:
        cols.append(col)

print(df_[cols + ['Class']].corr()['Class'])

#Relations between numerical columns [cols] and the target 'Class'

#=======================================
# EDA -- visualization 
#=======================================

plt.figure(figsize=(9,6))
sns.countplot(x='Class',data=df_)
plt.title('Class distribution (0=No, 1=Yes)')
plt.show()

#=======================================
# Scaling
#=======================================
df=df_.copy()

x=df.drop('Class',axis=1).values
y=df['Class'].values

#----------Split-------------
scaler=RobustScaler()
x_scaled= scaler.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=42)


smote = SMOTE(random_state=42)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

#To Make Balance in messy data values 

#=======================================
# Features Selection --
#=======================================

select=RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1)
select.fit(x_train_res,y_train_res)

importances=pd.Series(select.feature_importances_,index=cols).sort_values(ascending=False)
print(importances.head(10))

features=importances.head(10).index.to_list()
print(features)

#=======================================
# Choosing Model
#=======================================

models={

    'Logistic':LogisticRegression(max_iter=200),
    'SVM':SVC(kernel='rbf',gamma='scale',random_state=42),
    'Ridge':RidgeClassifier(random_state=42,max_iter=200),
    'Gradient':GradientBoostingClassifier(random_state=42),
    'DecisionTree':DecisionTreeClassifier(random_state=42,max_depth=5),
    'Perceptron':Perceptron(random_state=42,eta0=0.1)

}

choosen=[]

for name ,model in models.items():
    
    model.fit(x_train_res,y_train_res)
    pred=model.predict(x_test)
    f1=f1_score(y_test,pred)
    cm=confusion_matrix(y_test,pred)
    choosen.append({'Model':name,'F1':f1,'CM':cm})

best_df=pd.DataFrame(choosen).sort_values(by='F1',ascending=False)

print(best_df)



#-- best model is Logistic

#Applying the best Model

model=LogisticRegression(max_iter=200)
model.fit(x_train_res,y_train_res)
y_pred=model.predict(x_test)
y_prob=model.predict_proba(x_test)[:,1]
acc=model.score(x_test,y_test)
print("Accuracy",acc)

plt.scatter(x_scaled[:,0],x_scaled[:,1],c=y,cmap='coolwarm')
plt.title("True Data ")
plt.show()

#==============================================
# Visualization --
#==============================================

#------------Confusion Matrix--------------

cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#-----------Probabity Curve----------------
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0,1],[0,1],'k--')
plt.title('ROC Curve - Logistic Regression ')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()

#-------Relations of All Features-----------

corr = df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

#============================================
# Saving The Model
#============================================

joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model, 'best_model_.pkl')
print("Saved scaler and model.")

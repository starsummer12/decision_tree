import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings("ignore")



data=pd.read_csv("/Users/fireyr/Documents/data/ml_data/tt/train.csv")
data=data[['Survived', 'Pclass','Sex', 'Age', 'SibSp','Parch', 'Fare','Embarked']]
data['Age']=data['Age'].fillna(data['Age'].mean())
data.fillna(0,inplace=True)

data['Sex']=[1 if x=='male' else 0 for x in data['Sex']]

data['p1']=np.array(data['Pclass']==1).astype(np.int32)
data['p2']=np.array(data['Pclass']==2).astype(np.int32)
data['p3']=np.array(data['Pclass']==3).astype(np.int32)

del data['Pclass']
data.Embarked.unique()

data['e1']=np.array(data['Embarked']=='S').astype(np.int32)
data['e2']=np.array(data['Embarked']=='C').astype(np.int32)
data['e3']=np.array(data['Embarked']=='Q').astype(np.int32)

del data['Embarked']

data.values.dtype
data_train=data[['Sex', 'Age','SibSp','Parch', 'Fare','p1','p2','p3','e1','e2','e3']].values
data_target=data['Survived'].values.reshape(len(data),1)

x_train,x_test,y_train,y_test=train_test_split(data_train,data_target,test_size=0.2)
model=DecisionTreeClassifier()
model.fit(x_train,y_train)

# def m_score(depth):
#     model=DecisionTreeClassifier(max_depth=depth)
#     model.fit(x_train,y_train)
#     train_score=model.score(x_train,y_train)
#     test_score=model.score(x_test,y_test)
#     return train_score,test_score
#
# depths=range(2,15)
# scores=[m_score(depth) for depth in depths]


# train_s=[s[0] for s in scores]
# test_s=[s[1] for s in scores]
#
# plt.plot(train_s)
# plt.plot(test_s)
# plt.show()


# def m_score(value):
#     model=DecisionTreeClassifier(min_impurity_split=value)
#     model.fit(x_train,y_train)
#     train_score=model.score(x_train,y_train)
#     test_score=model.score(x_test,y_test)
#     return train_score,test_score
#
# values=np.linspace(0,0.5,50)
# scores=[m_score(value) for value in values]
# train_s=[s[0] for s in scores]
# test_s=[s[1] for s in scores]
#
# best_index=np.argmax(test_s)
# best_score=test_s[best_index]
# best_value=values[best_index]
#
# plt.plot(train_s)
# plt.plot(test_s)
# plt.show()


values=np.linspace(0,0.5,50)
depths=range(2,15)
param_grid={'max_depth':depths,'min_impurity_split':values}
model=GridSearchCV(DecisionTreeClassifier(),param_grid,cv=5)
model.fit(data_train,data_target)
print(model.best_params_)
print(model.best_score_)



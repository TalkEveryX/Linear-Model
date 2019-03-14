
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
#数据获取
boston=load_boston()
x=boston.data
y=boston.target
boston.keys()
boston_pd=pd.DataFrame(boston['data'],columns=boston.feature_names)
boston_pd['Target']=pd.DataFrame(boston['target'],columns=['Target'])

#查看数据
boston_pd.head(3)

#特征选择
boston_pd.corr().sort_values(by=['Target'],ascending=False)
sns.set( palette="muted", color_codes=True)
sns.pairplot(boston_pd , vars=[ 'RM', 'Target'])
x=x[y<50]
y=y[y<50]
print(x.shape)
print(y.shape)

#打乱并分割训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=10)
print("Training and testing split was successful.")

#模型拟合
model=LinearRegression()
model.fit(X_train,y_train)

#模型测试
y_pred = model.predict(X_test)

#使用R2作为模型评分标准
score = r2_score(y_test, y_pred)
print("The R2 score is ",score)

#打印模型参数
print(model.coef_)

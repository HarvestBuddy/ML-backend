import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

df=pd.read_csv("/content/crop_yield.csv")
df.isnull().sum()
count =0
dicti={}
names=df['Crop'].unique()
for name in names:
  if name not in dicti:
    dicti[name]=count
    count+=1
print(dicti)
df['Crop']=df['Crop'].map(dicti)
df['Season']
print(df['Season'].unique())
dict2={'Whole Year ':0 ,'Kharif     ':1 ,'Rabi       ':2, 'Autumn     ':3, 'Summer     ':4,'Winter     ':5}
df['Season']=df['Season'].map(dict2)
df.dtypes
count =0
dicti3={}
names=df['State'].unique()
for name in names:
  if name not in dicti:
    dicti3[name]=count
    count+=1
    # print(name,count)
df['State']=df['State'].map(dicti3)
df.columns
x=df[['Crop', 'Crop_Year', 'Season', 'State', 'Area','Annual_Rainfall', 'Fertilizer', 'Pesticide']]
y=df['Production']


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 42)

regressor.fit(x_train, y_train)
y_pred=regressor.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
print("decision tree",mae)
import pickle
with open("model_pickle","wb") as f:
  pickle.dump(regressor,f)
with open("model_pickle","rb") as f:
  mp= pickle.load(f)
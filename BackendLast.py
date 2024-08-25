import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression ,Ridge,Lasso,ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib
import streamlit as st
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r'tehranhouses.csv')

def lower_upper(x) :
    Q1 = np.percentile(x,25)
    Q3 = np.percentile(x,75)
    IQR = Q3-Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    return lower,upper
lower_area , upper_area = lower_upper( df['Area'] )
lower_price , upper_price = lower_upper( df['Price'] )

area_outliers = np.where(df[ (df.Area < lower_area) |(df.Area > upper_area) ])
price_outliers = np.where(df[ (df.Area < upper_price) |(df.Area > upper_price) ])

df.drop( df['Area'] == area_outliers )
df.drop( df['Price']== price_outliers )

df.dropna(inplace=True)
df_final = pd.get_dummies(df['Address'],dtype=int)
df_final = df.merge(df_final,left_index=True,right_index=True)
df_final.drop(columns = 'Address', inplace = True)

boolean_features = ['Parking','Warehouse','Elevator']
df_final[boolean_features] =df_final[boolean_features].astype('int64')

df_final['Area'] = df_final['Area'].apply(lambda x : re.sub('[^\d]','',x))
df_final['Area'] = pd.to_numeric(df_final['Area'],errors='coerce')
if '3,600' in df_final['Price'] :
    df_final = df_final.drop(df_final[df_final['Price']=='3,600'].index)

y = df_final['Price']
df_final.drop(columns=['Price','Price(USD)'],inplace=True)
x = df_final

columns =x.columns
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

def parameters_finder(model,parameters) :

    grid = GridSearchCV(model , param_grid=parameters , refit=True ,cv = KFold(shuffle=True,random_state=1) , n_jobs= -1)
    grid_fit = grid.fit(x_train,y_train)    
    return grid_fit

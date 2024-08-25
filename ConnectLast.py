import streamlit as st
from backend2.ipynb import columns,x_train,y_train,parameters_finder
#from projectlas4 import df
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression ,Ridge,Lasso,ElasticNet
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

area = st.number_input("input area" , 0.0)
room = st.number_input("input number of room" , 0.0)
parking = st.selectbox("is parking",['true','false'])
Warehouse = st.selectbox("is Warehouse",['true','false'])
Elevator = st.selectbox("is Elevator",['true','false'])
address = st.text_input("input address","zaefereniye")

def predict() :
    if address in columns :
      row = np.array(columns)
      x = pd.DataFrame([row] ,columns= columns)

      x.iloc[:,0] = area
      x.iloc[:,1] = room
      if parking == 'true' :
         x.iloc[:,2] = True
      else :
         x.iloc[:,2] = False   
      if Warehouse == 'true' :
         x.iloc[:,3] = True
      else :
         x.iloc[:,3] = False   
      if Elevator == 'true' :
         x.iloc[:,4] = True 
      else :
        x.iloc[:,4] = False      
      x.iloc[: , 5 : 197] = 0
      x.loc[:,[address]] = 1

      eln = ElasticNet(random_state = 1) # Linear regression with combined L1 and L2 priors as regularizer.
      param_eln = {'alpha': [0.001, 0.01, 0.1, 1, 10],
            'l1_ratio': [0.3, 0.4, 0.5, 0.6, 0.7]}
      grid_fit = parameters_finder(eln, param_eln)
      prediction = grid_fit.predict(x)       

      st.success('{:,.2f}'.format(prediction[0]))
      
    else :
      st.error("address isnot in list please input address with Big character in first")   
st.button('predict/result',on_click = predict)

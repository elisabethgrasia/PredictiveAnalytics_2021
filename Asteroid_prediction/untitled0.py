# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 17:12:25 2021

@author: alvak
"""

import numpy as np
import pandas as pd
import os

Asteroid= pd.read_csv('Asteroid_Updated.csv')
df = Asteroid
temp = df

aaa=pd.isnull(df['name'])
name_indices_nan = np.where(np.asarray(aaa)==True)[0]
processed_dataframe= temp.drop(name_indices_nan)

print("Nan in columns after removing nan in names \n",processed_dataframe.isnull().sum())


for column in processed_dataframe:
    if processed_dataframe[column].isnull().sum() > 5000:
        processed_dataframe= processed_dataframe.drop([column], axis=1)
        
        
print("Nan in columns after removing some columns \n",processed_dataframe.isnull().sum())
        

for column in processed_dataframe:
    if processed_dataframe[column].isnull().any():
        aaa=pd.isnull(processed_dataframe[column])
        indices_nan = np.where(np.asarray(aaa)==True)[0]
        processed_dataframe = processed_dataframe.reset_index(drop=True)
        processed_dataframe= processed_dataframe.drop(indices_nan)


print("Nan in columns after removing rows \n",processed_dataframe.isnull().sum())

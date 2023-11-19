#%% LOAD
#-*- coding: utf-8 -*-
"""
Created on Fri Nov 17 15:35:34 2023

"""
import numpy as np
import pandas as pd
import statistics as st
import matplotlib.pyplot as plt

filepaths = [r'C:\Users\sgast\Downloads\datathon-mibanco\test.csv', 
             r'C:\Users\sgast\Downloads\datathon-mibanco\train.csv',
             r'C:\Users\sgast\Downloads\datathon-mibanco\customers.csv',
             r'C:\Users\sgast\Downloads\datathon-mibanco\balances.csv']
df_balances = pd.read_csv(filepaths[3])
df_customers = pd.read_csv(filepaths[2])
df_train = pd.read_csv(filepaths[1])
df_test = pd.read_csv(filepaths[0])

#%%SCALER DEF

def SCALER(df:pd.DataFrame):
    cuanti = []
    for i in df.columns:
        if df[i].dtype in ['float64','int64']:
            cuanti.append(i)

    from sklearn.preprocessing import StandardScaler
    _scaler_ = StandardScaler()
    df[cuanti] = _scaler_.fit_transform(df[cuanti])

#%% CLEANING CUSTOMERS

"""
    cleaning customers
"""
df_customers['BOOL_VIVIENDA'] = ((
    df_customers.NO_DEPARTAMENTO.notnull()
    ) & (
        df_customers.NO_PROVINCIA.notnull()
        )
        )

#df_customers['BOOL_CIIU'] = (
#    df_customers.DE_CIIU.str.find(
#        'OTRAS ACTIVIDADES NO CLASIFICAD.EN OTRA PARTE')==-1)
        
df_customers.drop(columns= ['NO_DEPARTAMENTO','NO_PROVINCIA', 'DE_CIIU'],
                  axis = 1,
                  inplace = True)

df_customers.EDAD.fillna(
    df_customers.EDAD.mean(), inplace=True)

df_customers = pd.get_dummies(
    data = df_customers,
    columns = ['CO_TIPO_SEXO'])
     
#df_customers['YEAR'] = df_customers['PER_BANCARIZACION']//100
#df_customers['MONTH'] = df_customers['PER_BANCARIZACION']%100

#df_customers.drop('PER_BANCARIZACION',axis = 1, inplace = True)
   
""" 
    *********** CUSTOMERS CLEANED *********** 
"""

#%% AGRUPANDO SALDOS NO CORRER
saldos_dola = []

saldos = []

for i in df_balances.columns:
    if i.find('SALDO')!=-1:
        if i.find('DOLA')!=-1:
            saldos_dola.append(i)
        else:
            saldos.append(i)
df_balances['BOOL_SALDO'] = (
    df_balances[saldos].sum(axis = 1) 
    + 4*df_balances[saldos_dola].sum(axis=1)
    >= 10_000
    )

df_balances.drop(
    columns = saldos_dola + saldos, 
    axis = 1, 
    inplace = True)

#%% X CREATION

df_balances.drop(columns = ['PERIODO'], 
      axis = 1,
      inplace=True)
        
X=(
    df_balances
    .groupby('ID')
    .agg([
        np.min,np.max,np.mean,np.std
        ])
    .reset_index()
)

X.columns=range(len(X.columns))
X.rename(columns={0:'ID'},inplace=True)

X = X.merge(
    df_customers,
    on = 'ID',how='inner')
X.columns = X.columns.astype(str)

#%%

SCALER(df = X)


#%% ADDING TRAINING TO X
X=pd.merge(
    X,
    df_train,
    on='ID',
    how='left'
    )

X.columns = X.columns.astype(str)

#%%TRAINING WITH LOGISTIC

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X[
    X.
    TARGET.
    notnull()]
    .drop(['ID','TARGET'], axis=1), 
    X[
    X.
    TARGET.
    notnull()]
    .TARGET)

#%%

X.loc[
    X.TARGET.isnull(),
    'TARGET'] = model.predict(
        X[
            X.
            TARGET.
            isnull()].
        drop(['ID','TARGET'],axis=1))

(X[
    X
    .ID
    .isin(
        df_test.ID)]
    [['ID','TARGET']]
    .to_csv('test.csv',index=False)
)



# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 17:10:10 2023

@author: sgast
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv")
df.head()

#%%
""" 
   we select all floats and ints data
"""

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

#%%

"""
    Adding smth new: plotting each column vs a 
    given column (in this case, ENGINSEIZE)
    instead of plotting each one each time.
    ofc, colour package is just for visibility
"""

from colour import Color
colors = list(Color("black").range_to(Color("blue"),4))
vs = 'ENGINESIZE'

for i in list(cdf):
    if i!=vs:
        cdf.plot.scatter(
            x = i,
            y = vs,
            s = 5,
            color = str(colors[list(cdf).index(i)]))
plt.show()    

#%%

"""
    now, let's partition our data.
"""

msk = np.random.rand(len(df)) < 0.8 
train = cdf[msk]
test = cdf[~msk]

"""
    what does it do? msk becomes an array of 
    lenght = len(df) random numbers from 0 to 1.
    adding "<0.8" makes it boolean, in the obvious sense
    cdf[msk] makes cdf take a boolean array as a filter
    so we have choosen randomly msk.sum() rows of cdf
    and we'll test with len(df)-msk.sum() rows
"""
#%%

"""
    we R training:
"""

from sklearn import linear_model

regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
print ('Coefficients: ', regr.coef_)
print ('Intercept: ', regr.intercept_)

from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
predicted_y = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y - predicted_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y - predicted_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , predicted_y))
#%%

"""
    now lets just avoid the previous cells and implement 
    a function wich prints needed information to predict.
    the idea is a map 
        dataframe, test_size, column_to_predict 
        :\mapsto: 
        predicted_y, 
        error of predicted value related to original,
        coefficients
    and do this column per columns, properly plotted
    the idea is to return a dictionary with key information
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
from colour import Color


#for now, the dataframe has to be already filtered with float values
def test_per_column(df, t_size, c_pred):
    
    #choosing columns we're working w
    col_list_to_test = list(df)
    col_list_to_test.remove(c_pred)
    
    #color for plot visuals
    colors = list(Color("black").range_to(Color("blue"),
                                          len(col_list_to_test)))
    
    #partition
    msk = np.random.rand(len(df)) < t_size 
    train = df[msk]
    test = df[~msk]
    
    #definition of variables
    regr_list = {}
    TRAIN_X = {}
    TEST_X = {}
    PRED_Y = {}
    train_y = np.asanyarray(train[[c_pred]])
    test_y = np.asanyarray(test[[c_pred]])
    final_dict = {}
    
    #filling numbers
    for i in col_list_to_test:
        regr_list[i] = linear_model.LinearRegression()
        TRAIN_X[i] = np.asanyarray(train[[i]])
        TEST_X[i] = np.asanyarray(test[[i]])
    
    #testing and almacenating results in a dictionary which will become 
    #a reasonable dataframe so we can keep only the good results
    for i in col_list_to_test:
        regr_list[i].fit(TRAIN_X[i], train_y)
        PRED_Y[i] = regr_list[i].predict(TEST_X[i])
        final_dict[i] = [ regr_list[i].coef_[0][0], regr_list[i].intercept_[0],r2_score(test_y , PRED_Y[i])]
    final_dict = pd.DataFrame(final_dict)
    final_dict = final_dict.T
    final_dict.columns = ['m', 'b', 'score']
    
    #we only taking into consideration top 3 r2_score values
    #and also only those whose r2 is >0.50    
    final_dict = (final_dict.
                  sort_values(by = 'score', 
                              ascending = True)
                  [final_dict.score>=0.5])
    #the rows which keep good results
    col_top = final_dict.head(3).index.tolist()
    
    #plotting top 3 columns
    for i in col_top:    
        plt.figure()
        plt.scatter(
            train[i], 
            train[c_pred], 
            c = str(colors[col_top.index(i)]), #color for visuals
            s = 1)
        plt.plot(
            TRAIN_X[i], 
            final_dict.loc[i,"m"]*TRAIN_X[i] + 
            final_dict.loc[i,"b"], 
            c = str(colors[col_top.index(i)]), #could be red as well
            label = f'${final_dict.loc[i,"m"]:.3f} x + {final_dict.loc[i,"b"]:.3f}$')
        plt.xlim(train[i].min()-1, 
                 train[i].max()+1)
        plt.xlabel(i)
        plt.ylabel(c_pred)
        plt.legend()
    plt.show()
    
    #returning what we looking for: the m,b and R indexed by the name of 
    #the top 3 columns 
    return final_dict

#%%
asd = test_per_column(df = cdf, 
                t_size = 0.75, 
                c_pred = 'CO2EMISSIONS')

"""
    the beauty of this codes lays in the fact that
    it works even if we had 100 columns 
    lets test it with another dataset
"""

#%%

df1 = pd.read_csv("C:\\Users\\sgast\\Downloads\\weatherHistory.csv\weatherHistory.csv")
cdf1 = df1.drop(['Formatted Date','Summary', 'Precip Type','Daily Summary'], axis = 1)

#%%

"""
    let's create a function wich clears a word and returns a lower case
    non parenthesis word 
"""

def cleaning_columns(a):
    special = ['(', ')']
    for i in special:
        a = a[:a.find(i)]
    a = a.replace(" ","_").lower()
    return a

cdf1.columns = [cleaning_columns(_) for _ in cdf1.columns]

#%%
cpred = 'temperature'
qwe = test_per_column(df = cdf1,
                t_size = 0.80, 
                c_pred = cpred)








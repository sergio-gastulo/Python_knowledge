
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 10:24:16 2023

@author: sgast
"""

import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta as td
import random as rd

"""
    As I am not able to show our sales, i have created a random 
    dataframe which, I hope, will asimilate to our real numbers. 
"""

start_date = dt(2000,1,1)
num_days = 30000
date_array = [start_date + td(days=_) for _ in range(num_days)]
#date_array = np.array([date_array]).flatten()
num_types_menu = 60
menu_limit = 70

"""
    As we only cook 3 menus on average per day, having filled 
    our dataframe with 60 menus per day will not match our work
    So, i need to create a function which will randomly assing only 3
    out of 60 menus per given day
"""
def random_choice():
    while True:
        a = rd.randint(1, 25)
        b = rd.randint(1, 25)
        c = rd.randint(1, 25)
        if a + b + c <= menu_limit and a+b+c >= 20:
            return [a,b,c]
def random_index():
    while True:
        a = rd.randint(1, num_types_menu)
        b = rd.randint(1, num_types_menu)
        if b != a:
            c = rd.randint(1, num_types_menu)
            if c != b and c!= a:
                return [a-1,b-1,c-1]

array_definitivo = np.zeros(shape = (num_days,num_types_menu), 
                            dtype= int)

for _ in range(num_days):
    array_definitivo[_][random_index()] = random_choice()

#%%

menu_list = ['menu'+str(_+1) for _ in range(num_types_menu)]
df = pd.DataFrame(array_definitivo)
df.columns = menu_list

df['Fecha'] = date_array 
df['Venta_diaria'] = df[menu_list].sum(axis=1)


#%%
import matplotlib.pyplot as plt
"""
    Plotting sales per day (Not recommended AT ALL)
"""
df.plot(x = 'Fecha', y = 'Venta_diaria', kind = 'line',color = 'black')
plt.xlabel('Día')
plt.ylabel('N° de ventas')
plt.title('Ventas diarias')
plt.grid(True)
plt.show()

#%%
"""
    Plotting sales per week,
    we create a new column called Semana
"""
df['Semana'] = df['Fecha'].dt.isocalendar().week

#La agrupamos por promedios
df.groupby(['Semana'])['Venta_diaria'].mean().plot(
    x = 'Semana', 
    y = 'Venta_diaria', 
    kind = 'line', 
    color = 'blue')
plt.xlabel('Semana')
plt.ylabel('N° de ventas promedio')
plt.grid(True)
plt.title('Ventas')
plt.show()

#%%
"""
    per month,
    called Mes
"""
df['Mes'] = df['Fecha'].dt.month
df.groupby(['Mes'])['Venta_diaria'].mean().plot(x = 'Semana', y = 'Venta_diaria', 
             kind = 'line', color = 'blue')
plt.xlabel('Semana')
plt.ylabel('N° de ventas promedio')
plt.grid(True)
plt.title('Ventas')
plt.show()

#%%
"""
    Presentaremos ananlisis por mes:
    Para ello, creamos un data frame que varia respecto
    al mes elegido
"""

import calendar
import datetime as dt

mes = 'August'
n_mes = list(calendar.month_name).index(mes)

df_mes = df[df['Mes'] == n_mes]
print(f'''\nPromedio de ventas en {mes}: 
      {round(df_mes["Venta_diaria"].mean(),2)}\n''')
print(f'''Venta mas baja en {mes}: 
      {df_mes["Venta_diaria"].min()}\n''')
print(f'''Dia de venta mas baja en {mes}: 
      {df_mes.loc[df_mes["Venta_diaria"].idxmin(),"Fecha"].date().strftime("%A, %B %d, %Y")}\n''')
print(f'''venta mas alta en {mes}: 
      {df_mes["Venta_diaria"].max()}\n''')
print(f'''Dia de venta mas alta en {mes}: 
      {df_mes.loc[df_mes["Venta_diaria"].idxmax(),"Fecha"].date().strftime("%A, %B %d, %Y")}\n''')

"""
    Valueable information:
    Jan: Tue, Sat
    Feb: Mon, Sat
    March: Thu, Sat
    Apr: Mon, Fri
    May: Thu, Fri
    Jun: Wed, Mond
    Jul: Sat, Sat
    August: Wed, Sat
"""


#%%
"""
    Plotting Days_of_the_week of each week:
    in the same plot
    NOT RECOMMENDED IF THERE IS TOO MUCH DATA
"""

from colour import Color
colors = list(Color("black").range_to(Color("blue"),6))

day_dict= {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday"
}

fig, ax = plt.subplots()

for i in range(7):
    ax.scatter(
        df[df.Fecha.dt.weekday == i].Semana,
        df[df.Fecha.dt.weekday == i].Venta_diaria,
        label = f'{day_dict.get(i)}',
        color = str(colors[i]))

ax.set_xlabel('Semana')
ax.set_ylabel('Venta por dia de semana')
ax.legend(loc = 'best')



#%%
"""
    On diferents plots:
"""

for i in range(7):
    df[df.Fecha.dt.weekday == i].plot(
        kind = 'scatter',
        x = 'Semana',
        y = 'Venta_diaria',
        color = 'blue',
        label = f'{day_dict.get(i)}')
plt.xlabel('Semana')
plt.ylabel('Venta dia de semana')
plt.legend()
plt.show()

"""
    This is worth in case we only working with a year,
    so to have an aceptable line plot, we need to group it
    by its weeknumber and its mean pear year 
    (monday1_2019 + monday1_2020+...)/n 
"""
#%%
df['Day'] = df.Fecha.dt.weekday
for i in range(7):
    df.groupby(['Semana','Day'])['Venta_diaria'].mean().reset_index()[df.Day == i].plot(
        #kind = 'scatter',
        x = 'Semana',
        y = 'Venta_diaria',
        color = 'blue',
        label = f'{day_dict.get(i)}')

plt.xlabel('Semana')
plt.ylabel('Venta dia de semana')
plt.legend()
plt.show()

#%%
"""
    Plotting Average of week vs 
    its minimum and its maximum
"""

df_week = df.groupby('Semana')[
    'Venta_diaria'].agg([np.min,np.mean,np.max,np.var])
df_week.columns = ['Min','Prom_sem','Max','Var']

fig,ax = plt.subplots()

ax.plot(list(df_week.index),
          df_week.Prom_sem, 
          color = 'black',
          label = 'Venta promedio')
ax.plot(list(df_week.index),
          df_week.Min, 
          color = 'red',
          label = 'Venta minima')
ax.plot(list(df_week.index),
          df_week.Max, 
          color = 'blue',
          label = 'Venta maxima')
ax.set_xlabel('Semana')
ax.set_ylabel('Venta')
ax.legend(loc = 'best')

#%%

"""
    Appending columns where min and max 
    days are attained per week
"""
list_min = []
list_max = []
for i in range(1,54):
    list_min.append(
            df.loc[df[df.Semana == i].Venta_diaria.idxmin(),'Fecha'].date())
    list_max.append(
            df.loc[df[df.Semana == i].Venta_diaria.idxmax(),'Fecha'].date())


df_week['Min_day'] = pd.to_datetime(list_min)
df_week['Max_day'] = pd.to_datetime(list_max)

#%%
df_week.Min_day.dt.strftime("%A").value_counts().plot(
    kind = 'bar',
    color = 'black', 
    grid = True,
    title = 'Histograma de las ventas por dia',
    fontsize = 12,
    rot = 0)

"""
    As the plot shows, the "worst" day is the first day shown 
"""

#%%
df_week.Max_day.dt.strftime("%A").value_counts().plot(
    kind = 'bar',
    color = 'black', 
    grid = True,
    title = 'Recuento de veces en las que el dia fue el mejor de la semana',
    fontsize = 12,
    rot = 0,
    legend = False,
    xlabel = 'Dias de venta')













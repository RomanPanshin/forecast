import numpy as np
import pandas as pd
import itertools
from datetime import datetime, date, time, timedelta
from IPython.display import display

# Import matplotlib, seaborn for visualization
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import mean_squared_error as MSE
from math import sqrt
import statsmodels.api as sm
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import the data
weather_data = pd.read_csv('weather_data_test.csv',parse_dates=['datetime'], sep=';', decimal=','
                     , infer_datetime_format=True)


# Select the datetime and the temperature columns
temp_df = weather_data[["datetime","T_mu"]]

mask = (temp_df['datetime'] >= '2001-01-01') & (temp_df['datetime'] <= '2019-05-21')
temp_df = temp_df.loc[mask]

# Reset the index
#temp_df.set_index("datetime", inplace=True)

#def predict(d):
    #prevYears = temp_df['datetime'] >= '2016-01-01';
    #prevDays =

temp_df['year'], temp_df['month'],temp_df['day'] = temp_df['datetime'].dt.year, temp_df['datetime'].dt.month, temp_df['datetime'].dt.day

#дата на которую вычисляем погоду
d = date(2019, 2, 14)

# vchera
prev_d = d - timedelta(days=1)
print (prev_d)
# vichilayem srednee na cvhera v proshlih godah
filter_prev_year = (temp_df['day'] == prev_d.day) & (temp_df['month'] == prev_d.month) & (temp_df['year']<prev_d.year)
filter_prev_year =  temp_df.loc[filter_prev_year]
display(filter_prev_year)
display(filter_prev_year['T_mu'].mean())




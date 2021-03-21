import numpy as np
import pandas as pd
import itertools
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

mask = (temp_df['datetime'] >= '2016-01-01') & (temp_df['datetime'] <= '2019-05-21')
temp_df = temp_df.loc[mask]

# Reset the index
temp_df.set_index("datetime", inplace=True)

def m3():
    predicted_df = temp_df["T_mu"].to_frame().shift(1).rename(columns = {"T_mu": "T_mu_pred" })
    actual_df = temp_df["T_mu"].to_frame().rename(columns = {"T_mu": "T_mu_actual" })

    # Concatenate the actual and predicted temperature
    one_step_df = pd.concat([actual_df,predicted_df],axis=1)

    # Select from the second row, because there is no prediction for today due to shifting.
    one_step_df = one_step_df[1:]

    # Fit the SARIMAX model using optimal parameters
    mod = sm.tsa.statespace.SARIMAX(one_step_df.T_mu_actual,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

    results = mod.fit()

    pred = results.get_prediction(start=pd.to_datetime('2017-05-19'), dynamic=False)
    pred_ci = pred.conf_int()
    ax = one_step_df.T_mu_actual['2015':].plot(label='observed')
    pred.predicted_mean.plot(ax=ax, label='Forecast')

    ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature (in Celsius)')
    plt.ylim([-20,30])
    plt.legend()
    #plt.show()
    y_forecasted = pred.predicted_mean
    y_truth = one_step_df.T_mu_actual['2017-05-19':]
    print(y_forecasted.shape)
    print(y_truth.shape)
    # Compute the mean square error
    mse = MSE(y_truth, y_forecasted, squared=True)
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
    
m3()

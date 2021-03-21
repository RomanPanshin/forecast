import numpy as np
import pandas as pd
from IPython.display import display

# Import matplotlib, seaborn for visualization
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import the data
weather_data = pd.read_csv('weather_data_test.csv',parse_dates=['datetime'], sep=';', decimal=','
                     , infer_datetime_format=True)

# Check the shape of the dataset
print(weather_data.shape)

# Select the datetime and the temperature columns
temp_df = weather_data[["datetime","T_mu"]]
display(temp_df.head(10))

mask = (temp_df['datetime'] >= '2016-01-01') & (temp_df['datetime'] <= '2019-05-21')
temp_df = temp_df.loc[mask]

# Reset the index
temp_df.set_index("datetime", inplace=True)

# Inspect first 5 rows and last 5 rows of the data
display(temp_df.head(5))
display(temp_df.tail(5))

display(temp_df.describe())

print(temp_df.loc[temp_df["T_mu"] == temp_df["T_mu"].max()])
print(temp_df.loc[temp_df["T_mu"] == temp_df["T_mu"].min()])

def m1():
    plt.figure(figsize=(16,10), dpi=100)
    plt.plot(temp_df.index, temp_df.T_mu, color='tab:red')
    plt.gca().set(title="Daily Temperature in Helsinki, Finland from 2016 to 2019", xlabel='Date', ylabel="Degree (in Celsius)")
    plt.show()

def m2():
    from statsmodels.tsa.seasonal import seasonal_decompose

    # Additive Decomposition
    result_add = seasonal_decompose(temp_df.T_mu, model='additive', extrapolate_trend='freq', freq=365)

    # Plot
    plt.rcParams.update({'figure.figsize': (10,10)})
    result_add.plot().suptitle('Additive Decomposition', fontsize=22)
    plt.show()

def m3():
    predicted_df = temp_df["T_mu"].to_frame().shift(1).rename(columns = {"T_mu": "T_mu_pred" })
    actual_df = temp_df["T_mu"].to_frame().rename(columns = {"T_mu": "T_mu_actual" })

    # Concatenate the actual and predicted temperature
    one_step_df = pd.concat([actual_df,predicted_df],axis=1)

    # Select from the second row, because there is no prediction for today due to shifting.
    one_step_df = one_step_df[1:]
    display(one_step_df.head(10))

#def m4():
    from sklearn.metrics import mean_squared_error as MSE
    from math import sqrt

    # Calculate the RMSE
    temp_pred_err = MSE(one_step_df.T_mu_actual, one_step_df.T_mu_pred, squared=False)
    print("The RMSE is",temp_pred_err)

#def m5():
    import itertools
    # Define the p, d and q parameters to take any value between 0 and 2
    p = d = q = range(0, 2)

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

#    print('Examples of parameter combinations for Seasonal ARIMA...')
#    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
#    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
 #   print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
 #   print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

# Import the statsmodels library for using SARIMAX model
    import statsmodels.api as sm

# Fit the SARIMAX model using optimal parameters
    mod = sm.tsa.statespace.SARIMAX(one_step_df.T_mu_actual,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

    results = mod.fit()
    results.plot_diagnostics(figsize=(15, 12)) ###
    plt.show() ##

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
    plt.show() ##
    y_forecasted = pred.predicted_mean
    y_truth = one_step_df.T_mu_actual['2017-05-19':]
    print(y_forecasted.shape)
    print(y_truth.shape)
    # Compute the mean square error
    mse = MSE(y_truth, y_forecasted, squared=True)
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
    
m3()

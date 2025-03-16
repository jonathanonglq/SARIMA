import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA as ARIMA
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import itertools
import math

def test_stationarity(ts, window = 12, xlabels = []):
    ma = ts.rolling(window=window).mean()
    mstd = ts.rolling(window=window).std()

    fig, ax = plt.subplots(figsize=(15,5))
    plt.title('Rolling Mean & Standard Deviation')

    orig = plt.plot(ts, color='blue',label='Original')
    mean = plt.plot(ma, color='orange', label='Rolling Mean')
    std = plt.plot(mstd, color='darkgreen', label = 'Rolling Std')
    plt.legend()
    plt.tight_layout()

    if len(xlabels):
        xticks = np.arange(0, len(xlabels), 1)
        ax.set_xticks(xticks, list(xlabels))
        plt.xticks(rotation=90)

    results = adfuller(ts, autolag='AIC')
    print('Dickey-Fuller Test Results:')
    print('Test Statistic: {0:.3f}'.format(results[0]))
    print('P-value: {0:.3f}'.format(results[1]))

def tsplot(ts, figsize=(15,10)):
    
    fig = plt.figure(figsize=figsize)
    layout = (3,1)
    ts_ax = plt.subplot2grid(layout, (0, 0))
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (2, 0))

    xticks = np.arange(0, 20, 1)
    xlabels = ['{0:.0f}'.format(x) for x in xticks]
    acf_ax.set_xticks(xticks, labels=xlabels)
    pacf_ax.set_xticks(xticks, labels=xlabels)
    
    ts.plot(ax=ts_ax)
    ts_ax.set_title('Time Series')
    smt.graphics.plot_acf(ts, ax=acf_ax)
    smt.graphics.plot_pacf(ts, ax=pacf_ax)
    plt.tight_layout()

def seasonal_decompose(y, figsize=(15,10), period = 12, model = 'multiplicative'):
    fig = sm.tsa.seasonal_decompose(y,period = period, model = model).plot()
    fig.set_size_inches(figsize)
    plt.tight_layout()
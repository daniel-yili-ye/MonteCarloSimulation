"""
Monte Carlo simulations are used to test various outcome possibilities. 
They are often used to assess risk of a give trading strategy with options or stocks.
This is a simulation for the 90 day returns of a given stock ticker and number of simulations based on stock data taken from 2000-01-01.

"""

import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm

def mc_simulation(ticker, simulations):
    data = pd.DataFrame()
    data[ticker] = wb.DataReader(ticker, data_source='yahoo', start='2000-1-1')['Adj Close']

    log_returns = np.log(1 + data.pct_change())

    u = log_returns.mean()

    var = log_returns.var()

    drift = u - (0.5 * var)

    stdev = log_returns.std()

    days = 90

    daily_returns = np.exp(drift.values + stdev.values * norm.ppf(np.random.rand(days, simulations)))
    data_S0 = data.iloc[-1]

    price_list = np.zeros_like(daily_returns)
    price_list[0] = data_S0

    for t in range(1, days):
        price_list[t] = price_list[t - 1] * daily_returns[t]
        
    plt.figure(figsize=(10,6))
    plt.title("90 Day Monte Carlo Simulation for " + ticker)
    plt.ylabel("Price (USD)")
    plt.xlabel("Time (Days)")
    plt.plot(price_list)

mc_simulation("AON", 100)
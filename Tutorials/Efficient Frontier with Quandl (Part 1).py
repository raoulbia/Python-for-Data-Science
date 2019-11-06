# import needed modules
import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# get adjusted closing prices of 5 selected companies with Quandl
quandl.ApiConfig.api_key = '6sHkfEoiqR4ykcsQkSnh'
selected = ['CNP', 'F', 'WMT', 'GE', 'TSLA']
selected = ['CNP', 'F']
selected_ls = [['CNP', 'F'], ['WMT', 'GE']]
data = quandl.get_table('WIKI/PRICES', ticker = selected,
                        qopts = { 'columns': ['date', 'ticker', 'adj_close'] },
                        date = { 'gte': '2014-1-1', 'lte': '2016-12-31' }, paginate=True)

# reorganise data pulled by setting date as index with
# columns of tickers and their corresponding adjusted prices
clean = data.set_index('date')
table = clean.pivot(columns='ticker')
# print(table.head())

# calculate daily returns of the stocks (looks similar to basis point estimates)
returns_daily = table.pct_change()
# print(returns_daily.head())

# calculate annual returns of the stocks
returns_annual = returns_daily.mean() * 250 # looks similar to mu_values
returns_annual = returns_annual.to_numpy()
# print(returns_annual)
# print(returns_annual.shape)

# get daily and covariance of returns of the stock
cov_daily = returns_daily.cov()
# print(cov_daily.shape)
# print(cov_daily)
cov_annual = cov_daily * 250
# print(cov_annual)

# empty lists to store returns, volatility and weights of imiginary portfolios
port_returns = []
port_volatility = []
stock_weights = []

# set the number of combinations for imaginary portfolios
for selected in selected_ls:
    num_assets = len(selected)
    num_portfolios = 50000

    # populate the empty lists with each portfolios returns,risk and weights
    for single_portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        returns = np.dot(weights, returns_annual)
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
        port_returns.append(returns)
        port_volatility.append(volatility)
        stock_weights.append(weights)

    # a dictionary for Returns and Risk values of each portfolio
    portfolio = {'Returns': port_returns,
                 'Volatility': port_volatility}

    # extend original dictionary to accomodate each ticker and weight in the portfolio
    for counter,symbol in enumerate(selected):
        portfolio[symbol+' Weight'] = [Weight[counter] for Weight in stock_weights]

    # make a nice dataframe of the extended dictionary
    df = pd.DataFrame(portfolio)

    # get better labels for desired arrangement of columns
    column_order = ['Returns', 'Volatility'] + [stock+' Weight' for stock in selected]

    # reorder dataframe columns
    df = df[column_order]
    print(df.head())
    break

# plot the efficient frontier with a scatter plot
plt.style.use('seaborn')
df.plot.scatter(x='Volatility', y='Returns', figsize=(10, 8), grid=True)
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier')
plt.show()
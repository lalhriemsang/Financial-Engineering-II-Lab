import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
from datetime import datetime
from scipy.misc import derivative
from scipy.stats import norm
from tabulate import tabulate
import warnings

def fx(C, x, K, t, T, r, sigma):
    if t == T:
        return max(0, x - K), max(0, K - x)

    d1 = ( np.log(x/K) + (r + 0.5 * sigma * sigma) * (T - t) ) / ( sigma * np.sqrt(T - t) )
    d2 = ( np.log(x/K) + (r - 0.5 * sigma * sigma) * (T - t) ) / ( sigma * np.sqrt(T - t) )

    call_price = x * norm.cdf(d1) - K * np.exp( -r * (T - t) ) * norm.cdf(d2)
    return C - call_price


def dfx(fx, var=0, point=[]):
    args = point[:]
    def wraps(x):
        args[var] = x
        return fx(*args)

    return derivative(wraps, point[var], dx = 1e-4)


def newton_method(epsilon, C, x, K, t, T, r, x0):
    itnum = 1
    root = 0

    while itnum <= 1000:
        denom = dfx(fx, 6, [C, x, K, t, T, r, x0])
        if denom == 0:
            return -1

        x1 = x0 - fx(C, x, K, t, T, r, x0)/denom
        if abs(x1 - x0) <= epsilon:
            root = x1
            break

        x0 = x1
        itnum += 1

    return root


def compare_volatility(file, opt_type, nse_opt):
    options_df = pd.read_csv(file)
    stocks_df = pd.read_csv('./nsedata1.csv')
    stocks_df['Date']= pd.to_datetime(stocks_df['Date'])

    historical_volatility, implied_volatility = [], []
    strike_price, maturity = [], []
    random.seed(53)
    
    table = []
    lim = 1
    for index, row in options_df.iterrows():
        coin_toss = random.random()
        bound = 0.15
        if opt_type == 'HDFCBANK.NS':
            bound = 1
        
        if coin_toss <= bound:
            d1 = datetime.strptime(row['Expiry  '],  '%d-%b-%Y')
            d2 = datetime.strptime(row['Date  '],  '%d-%b-%Y')
            delta = d1 - d2
            if int(row['Call Price']) != 0 and delta.days != 0 and not(np.isnan(row['Underlying Value  '])):
                sigma = newton_method(1e-6, row['Call Price'], row['Underlying Value  '], row['Strike Price  '], 0, delta.days/365, 0.05, 0.6)
                reduced_df = stocks_df[(stocks_df['Date'] >= d2) & (stocks_df['Date'] <= d1)]

                reduced_df = reduced_df.set_index('Date')
                reduced_df = reduced_df.pct_change()
                returns = reduced_df[nse_opt]
                x = returns.to_list()
                if len(x) == 0 or sigma == -1:
                    continue

                mean = np.nanmean(np.array(x))
                std = np.nanstd(np.array(x))
        
                historical_volatility.append(std * np.sqrt(252))
                implied_volatility.append(sigma)
                strike_price.append(row['Strike Price  '])
                maturity.append(delta.days)

                if lim <= 20:
                    table.append([lim, row['Call Price'], row['Underlying Value  '], delta.days, historical_volatility[-1], implied_volatility[-1]])
                    lim += 1


    print('*****************  For {}  *****************'.format(opt_type))
    print(tabulate(table, headers=['SI No.', 'Call Price', 'Stock Price (S0)', 'Maturity (in days)', 'Historical Volatility', 'Implied Volatility']))
    mpl.rcParams.update(mpl.rcParamsDefault)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(strike_price, maturity, historical_volatility, marker = '.', label = 'historical volatility')
    ax.scatter(strike_price, maturity, implied_volatility, marker = '.', label = 'implied volatility')
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Maturity (in days')
    ax.set_zlabel('Implied volatiltiy')
    ax.set_title('Comparison between historical and implied volatility - {}'.format(opt_type))
    ax.set_zlim(-0.03, 1.2)
    plt.legend(loc = 'upper left')
    plt.savefig('./Data_c/' + opt_type + '_3D.jpg')
    plt.close()

    plt.rcParams["figure.figsize"] = (13, 4)
    plt.subplot(1, 2, 1)
    plt.scatter(strike_price, implied_volatility, marker = '.', label = 'historical volatility')
    plt.scatter(strike_price, historical_volatility, marker = '.', label = 'implied volatility')
    plt.xlabel('Strike price')
    plt.ylabel('Volatility')
    plt.title('Volatility vs Strike Price for {}'.format(opt_type))
    plt.ylim(-0.03, 1.2)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(maturity, implied_volatility, marker = '.', label = 'historical volatility')
    plt.scatter(maturity, historical_volatility, marker = '.', label = 'implied volatility')
    plt.xlabel('Maturity (in days)')
    plt.ylabel('Volatility')
    plt.title('Volatility vs Maturity for {}'.format(opt_type))
    plt.legend()
    plt.ylim(-0.03, 1.2)
    plt.savefig('./Data_c/' + opt_type + '_volatility.jpg')
    plt.close()

    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.scatter(historical_volatility, implied_volatility, marker = '.')
    plt.xlabel('Historical Volatility')
    plt.ylabel('Implied Volatility')
    plt.title('Historical vs Implied volatility for {}'.format(opt_type))
    plt.ylim(-0.03, 1.2)
    plt.savefig('./Data_c/' + opt_type + '_volatility_comparison.jpg')
    plt.close()

warnings.filterwarnings("ignore")

files = ['NIFTYoptiondata', 'AsianPaints', 'HDFCBANK', 'SBIN', 'TCS', 'WIPRO']
opt_type = ['NIFTY', 'ASIANPAINTS', 'HDFCBANK', 'SBIN', 'TCS', 'WIPRO']
nse_data_cols = ['Nifty', 'ASIANPAINT.NS', 'HDFCBANK.NS', 'SBIN.NS', 'TCS.NS', 'WIPRO.NS']

for idx in range(len(files)):
    files[idx] = './' + files[idx] + '.csv'
    compare_volatility(files[idx], opt_type[idx], nse_data_cols[idx])
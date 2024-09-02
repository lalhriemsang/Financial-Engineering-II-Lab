import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import pandas as pd

# prices are sorted from oldest to latest in rows


def Normal(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (x - mu)**2)


def d(sign, s0, K, r, sigma, t):
    d1 = (np.log(s0 / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    if sign == "+":
        return d1
    else:
        return d2


def BSM(s0, K, r, sigma, t, T):
    tao = T - t
    d1 = d("+", s0, K, r, sigma, tao)
    d2 = d1 - sigma * np.sqrt(tao)
    N1 = integrate.quad(Normal, -np.inf, d1, args=(0, 1))
    N2 = integrate.quad(Normal, -np.inf, d2, args=(0, 1))
    C = s0 * N1[0] - K * np.exp(-r * tao) * N2[0]

    N1 = integrate.quad(Normal, -np.inf, -d1, args=(0, 1))
    N2 = integrate.quad(Normal, -np.inf, -d2, args=(0, 1))
    P = K * np.exp(-r * tao) * N2[0] - s0 * N1[0]
    return C, P


def q1(stock, days=30):
    if stock.lower() == "bsedata1":
        df = pd.read_csv("./bsedata1.csv")
    else:
        df = pd.read_csv("./nsedata1.csv")

    for column in df.columns:

        column_mean = df[column].mean()
        df[column] = df[column].fillna(column_mean)

    df_Months = df.iloc[-days:-1]

    returns = df_Months.pct_change().iloc[1:]

    # print(returns)
    voltalities = []

    for column in df.columns:

        voltality = np.sqrt(returns[column].var())
        voltalities.append(voltality*np.sqrt(252))

    return voltalities


def q2(stock, A, T=6):
    if stock.lower() == "bsedata1":
        df = pd.read_csv("./bsedata1.csv")
        sigmas = q1("bsedata1")
    else:
        df = pd.read_csv("./nsedata1.csv")
        sigmas = q1("nsedata1")

    for column in df.columns:

        column_mean = df[column].mean()
        df[column] = df[column].fillna(column_mean)

    s0 = df.iloc[-1, :]

    A = round(A, 1)

    call = []
    put = []

    print(f"K={A}xS0")
    for i in range(len(s0)):
        C, P = BSM(s0[i], A*s0[i], 0.05, sigmas[i], 0, T/12)

        call.append(C)
        put.append(P)
        print(df.columns[i], ":")
        print(f"Call: {C} \t\t\t Put: {P}")

    print(" ")

    return call, put


def q3(stock, index, call=True):

    sigmas = []
    # 1234 days ~ 42 months

    n = int(np.ceil(1234/30))
    # print(f"no. of months: {n}")

    if stock.lower() == "bsedata1":
        df = pd.read_csv("./bsedata1.csv")

        for i in range(1, n):
            sigma = q1("bsedata1", 30*i)
            sigmas.append(sigma[index])
    else:
        df = pd.read_csv("./nsedata1.csv")

        for i in range(1, n):
            sigma = q1("nsedata1", 30*i)
            sigmas.append(sigma[index])

    # print(len(sigmas))

    s0 = df.iloc[:, index].iloc[-1]

    fig, axs = plt.subplots(2, 6, figsize=(12*6, 8))

    for As in range(5, 16):
        stock_c0 = []
        stock_p0 = []
        A = round(As*0.1, 1)

        for sigma in sigmas:
            C, P = BSM(s0, A*s0, 0.05, sigma, 0, 6/12)
            stock_c0.append(C)
            stock_p0.append(P)

        Ts = np.arange(len(sigmas))

        if As >= 5 and As <= 10:
            i = As - 5
            if call == True:
                axs[0, i].plot(Ts, stock_c0)
                axs[0, i].set_title(f"European Call\nS0 vs T (months)\nA={A}")
            else:
                axs[0, i].plot(Ts, stock_p0)
                axs[0, i].set_title(f"European Put\nS0 vs T (months)\nA={A}")

        else:
            i = (As-1) % 5
            if call == True:
                axs[1, i].plot(Ts, stock_c0)
                axs[1, i].set_title(f"European Call\nS0 vs T (months)\nA={A}")
            else:
                axs[1, i].plot(Ts, stock_p0)
                axs[1, i].set_title(f"European Put\nS0 vs T (months)\nA={A}")

    axs[1, 5].plot(Ts, sigmas)
    axs[1, 5].set_title("sigmas vs T")
    plt.suptitle(f"{df.columns[index]}")
    plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(hspace=0.5)
    plt.show()


# Please uncomment to display plots

# 1)
# a)
# df = pd.read_csv("./bsedata1.csv")
# columns = df.columns
# print("Volatilities")
# print("\t\tBSE:")
# volatilities = q1("bsedata1")
# for i in range(len(volatilities)):
#     print(columns[i], ":\t\t", volatilities[i])
# print(" ")
# df = pd.read_csv("./nsedata1.csv")
# columns = df.columns
# print("\t\tNSE:")
# volatilities = q1("nsedata1")
# for i in range(len(volatilities)):
#     print(columns[i], ":\t\t", volatilities[i])
# print(" ")

# b)
# print("BSE:")
# for A in np.arange(0.5, 1.6, 0.1):
#     q2("bsedata1", A)
# print("NSE:")
# for A in np.arange(0.5, 1.6, 0.1):
#     q2("nsedata1", A)

# c)
# for i in range(21):
#     q3("bsedata1", i)  # call
# for i in range(21):
#     q3("bsedata1", i, False)  # put

# for i in range(21):
#     q3("nsedata1", i)  # call
for i in range(1):
    q3("nsedata1", i, False)  # put

import numpy as np
import matplotlib.pyplot as plt
import math
import time


def americanPayoff(i, j, s0, K, N, u, d, r, delt, p, memPayoff, call):

    if i+j == N:
        if call:
            if ((i, j) in memPayoff) == False:
                memPayoff[(i, j)] = max(s0-K, 0)
            return memPayoff[(i, j)]
        else:
            if ((i, j) in memPayoff) == False:
                memPayoff[(i, j)] = max(K-s0, 0)
            return memPayoff[(i, j)]

    if call and ((i, j) in memPayoff):
        return memPayoff[(i, j)]
    elif call == False and ((i, j) in memPayoff):
        return memPayoff[(i, j)]

    if call:
        memPayoff[(i, j)] = max(s0-K, (americanPayoff(i+1, j, u*s0, K, N, u, d, r, delt, p, memPayoff, call) * p +
                                       americanPayoff(i, j+1, d*s0, K, N, u, d, r, delt, p, memPayoff, call)*(1-p))*np.exp(-r*delt))

        return memPayoff[(i, j)]
    else:
        memPayoff[(i, j)] = max(K-s0, (americanPayoff(i+1, j, u*s0, K, N, u, d, r, delt, p, memPayoff, call) * p +
                                       americanPayoff(i, j+1, d*s0, K, N, u, d, r, delt, p, memPayoff, call)*(1-p))*np.exp(-r*delt))
        return memPayoff[(i, j)]


def AmericanOption(s0, K, N, T, r, sig, optn="call"):
    delt = T / N
    u = np.exp(sig * delt**0.5 + (r - 0.5 * sig**2) * delt)
    d = np.exp(-sig * delt**0.5 + (r - 0.5 * sig**2) * delt)

    if d >= np.exp(r * delt) or np.exp(r * delt) >= u:
        print("Arbitrage Possible!")
        return

    delt = T / N
    p = (np.exp(r * delt) - d) / (u - d)

    memPayoff = {}

    if optn.lower() == "call":
        return americanPayoff(0, 0, s0, K, N, u, d, r, delt, p, memPayoff, True)
    else:
        return americanPayoff(0, 0, s0, K, N, u, d, r, delt, p, memPayoff, False)


s0 = 100
K = 100
M = 100
T = 1
r = 0.08
sig = 0.3
# print("M =", M)
# optionPrice = AmericanOption(s0, K, M, T, r, sig, "call")
# print("Call Price:", optionPrice)
# optionPrice = AmericanOption(s0, K, M, T, r, sig, "put")
# print("Put Price:", optionPrice)

# a)
# K = 95
# K = 105
# s = np.arange(50, 150, 1)
# call = []
# put = []

# for K in s:
#     call.append(AmericanOption(s0, K, M, T, r, sig, "call"))
#     put.append(AmericanOption(s0, K, M, T, r, sig, "put"))

# plt.plot(s, call, linewidth=1, label='call')
# plt.plot(s, put, linewidth=1, label='put')
# plt.xlabel("s")
# plt.ylabel("Option Price")
# plt.legend()
# plt.show()

# b)
# k = np.arange(50, 150, 1)
# call = []
# put = []

# for K in k:
#     call.append(AmericanOption(s0, K, M, T, r, sig, "call"))
#     put.append(AmericanOption(s0, K, M, T, r, sig, "put"))

# plt.plot(k, call, linewidth=1, label='call')
# plt.plot(k, put, linewidth=1, label='put')
# plt.xlabel("K")
# plt.ylabel("Option Price")
# plt.legend()
# plt.show()

# c)
# K = 95
# K = 105
# R = np.arange(0.01, 1, 0.001)
# call = []
# put = []

# for r in R:
#     call.append(AmericanOption(s0, K, M, T, r, sig, "call"))
#     put.append(AmericanOption(s0, K, M, T, r, sig, "put"))

# plt.plot(R, call, linewidth=1, label='call')
# plt.plot(R, put, linewidth=1, label='put')
# plt.xlabel("r")
# plt.ylabel("Option Price")
# plt.legend()
# plt.show()

# d)
# K = 95
# K = 105
# Sig = np.arange(0.1, 1, 0.01)
# call = []
# put = []

# for sig in Sig:
#     call.append(AmericanOption(s0, K, M, T, r, sig, "call"))
#     put.append(AmericanOption(s0, K, M, T, r, sig, "put"))

# plt.plot(Sig, call, linewidth=1, label='call')
# plt.plot(Sig, put, linewidth=1, label='put')
# plt.xlabel("sig")
# plt.ylabel("Option Price")
# plt.legend()
# plt.show()


# e)
# K=100
# K = 95
# K = 105
# m = np.arange(50, 120, 1)
# call = []
# put = []

# for M in m:
#     call.append(AmericanOption(s0, K, M, T, r, sig, "call"))
#     put.append(AmericanOption(s0, K, M, T, r, sig, "put"))

# plt.plot(m, call, linewidth=1, label='call')
# plt.plot(m, put, linewidth=1, label='put')
# plt.xlabel("M")
# plt.ylabel("Option Price")
# plt.legend()
# plt.show()

# 2)


def payOff(h, maxPrice, s0, u, d, p, r, delt, N, tabulate, table):

    if h == N:
        if tabulate == True:
            table[h].append(max(maxPrice-s0, 0))
        return max(maxPrice-s0, 0)

    payOffVal = (payOff(h+1, max(maxPrice, s0*u), s0*u, u, d, p, r, delt, N, tabulate, table)*p +
                 payOff(h+1, max(maxPrice, s0*d), s0*d, u, d, p, r, delt, N, tabulate, table)*(1-p))*np.exp(-r*delt)

    if tabulate == True:
        table[h].append(payOffVal)

    return payOffVal


def lookBackEuropeanOption(s0, T, N, r, sig, tabulate, table):
    delt = T/N
    u = np.exp(sig*delt**0.5 + (r-0.5*sig**2)*delt)
    d = np.exp(-sig*delt**0.5 + (r-0.5*sig**2)*delt)
    if d >= np.exp(r*delt) or np.exp(r*delt) >= u:
        print("Arbitrage Possible!")
        return

    p = (np.exp(r*delt) - d)/(u-d)

    return payOff(0, s0, s0, u, d, p, r, delt, N, tabulate, table)


s0 = 100
T = 1
R = 0.08
sig = 0.30

table = [[], [], [], [], [], []]

# a)
# for M in [5, 10, 25]:
#     print(f"M={M}")
#     start = time.time()
#     callPrice = lookBackEuropeanOption(
#         s0, T, M, R, sig, tabulate=False, table=table)
#     end = time.time()
#     print("time:", end-start)
#     print(callPrice)

# M=50
# print("M=50")
# start = time.time()
# callPrice = lookBackEuropeanOption(
#         s0, T, M, R, sig, tabulate=False, table=table)
# end = time.time()
# print("time:", end-start)
# print(callPrice)

# c)
# lookBackEuropeanOption(s0, T, N=5, r=R, sig=sig, tabulate=True, table=table)
# print(table)

# 3)


def dpLookBackPayOff(i, j, maxPrice, s0, u, d, p, r, delt, N, tabulate, table, memPayoff):

    if i+j == N:
        if ((i, j, maxPrice) in memPayoff) == False:
            memPayoff[(i, j, maxPrice)] = max(maxPrice-s0, 0)

        return memPayoff[(i, j, maxPrice)]

    if ((i, j, maxPrice) in memPayoff) == True:
        return memPayoff[(i, j, maxPrice)]

    memPayoff[(i, j, maxPrice)] = (dpLookBackPayOff(i+1, j, max(maxPrice, s0*u), s0*u, u, d, p, r, delt, N, tabulate, table, memPayoff)*p +
                                   dpLookBackPayOff(i, j+1, max(maxPrice, s0*d), s0*d, u, d, p, r, delt, N, tabulate, table, memPayoff)*(1-p))*np.exp(-r*delt)

    if tabulate == True:
        table[i+j].append(memPayoff[(i, j, maxPrice)])

    return memPayoff[(i, j, maxPrice)]


def dpLookBackEuropeanOption(s0, T, N, r, sig, tabulate, table):
    delt = T/N
    u = np.exp(sig*delt**0.5 + (r-0.5*sig**2)*delt)
    d = np.exp(-sig*delt**0.5 + (r-0.5*sig**2)*delt)
    if d >= np.exp(r*delt) or np.exp(r*delt) >= u:
        print("Arbitrage Possible!")
        return

    p = (np.exp(r*delt) - d)/(u-d)
    memPayoff = {}

    return dpLookBackPayOff(0, 0, s0, s0, u, d, p, r, delt, N, tabulate, table, memPayoff)


s0 = 100
T = 1
R = 0.08
sig = 0.30

table = [[], [], [], [], [], []]

# for M in [5,10,25,50]:
#     print("M:", M)
#     start = time.time()
#     callPrice = dpLookBackEuropeanOption(
#         s0, T, N=M, r=R, sig=sig, tabulate=False, table=table)
#     end = time.time()
#     print("time:", end-start)
#     print(callPrice)

# 4)


def dpNormalPayOff(i, j, K, s0, u, d, p, r, delt, N, tabulate, table, memPayoff):

    if i+j == N:
        if ((i, j) in memPayoff) == False:
            memPayoff[(i, j)] = max(s0-K, 0)
        return memPayoff[(i, j)]

    if ((i, j) in memPayoff) == True:
        return memPayoff[(i, j)]

    memPayoff[(i, j)] = (dpNormalPayOff(i+1, j, K, s0*u, u, d, p, r, delt, N, tabulate, table, memPayoff)*p +
                         dpNormalPayOff(i, j+1, K, s0*d, u, d, p, r, delt, N, tabulate, table, memPayoff)*(1-p))*np.exp(-r*delt)

    if tabulate == True:
        table[i+j].append(memPayoff[(i, j)])

    return memPayoff[(i, j)]


def dpEuropeanOption(s0, T, K, N, r, sig, tabulate, table):
    delt = T/N
    u = np.exp(sig*delt**0.5 + (r-0.5*sig**2)*delt)
    d = np.exp(-sig*delt**0.5 + (r-0.5*sig**2)*delt)
    if d >= np.exp(r*delt) or np.exp(r*delt) >= u:
        print("Arbitrage Possible!")
        return

    p = (np.exp(r*delt) - d)/(u-d)
    memPayoff = {}

    return dpNormalPayOff(0, 0, K, s0, u, d, p, r, delt, N, tabulate, table, memPayoff)


def NormalPayOff(i, j, K, s0, u, d, p, r, delt, N, tabulate, table):

    if i+j == N:
        return max(s0-K, 0)

    return (NormalPayOff(i+1, j, K, s0*u, u, d, p, r, delt, N, tabulate, table)*p +
            NormalPayOff(i, j+1, K, s0*d, u, d, p, r, delt, N, tabulate, table)*(1-p))*np.exp(-r*delt)


def EuropeanOption(s0, T, K, N, r, sig, tabulate, table):
    delt = T/N
    u = np.exp(sig*np.sqrt(delt) + (r-0.5*(sig**2))*delt)
    d = np.exp(-sig*np.sqrt(delt) + (r-0.5*(sig**2))*delt)
    if d >= np.exp(r*delt) or np.exp(r*delt) >= u:
        print("Arbitrage Possible!")
        return

    p = (np.exp(r*delt) - d)/(u-d)

    return NormalPayOff(0, 0, K, s0, u, d, p, r, delt, N, tabulate, table)


s0 = 100
K = 100
T = 1
R = 0.08
sig = 0.30

table = [[], [], [], [], []]

# for M in [5, 10, 25]:
#     print("M =", M)
#     start = time.time()
#     callPrice = EuropeanOption(
#         s0, T, K, N=M, r=R, sig=sig, tabulate=False, table=table)
#     end = time.time()
#     print("unoptimized time:", end-start)
#     print(callPrice)

#     start = time.time()
#     callPrice = dpEuropeanOption(
#         s0, T, K, N=M, r=R, sig=sig, tabulate=False, table=table)
#     end = time.time()
#     print("optimized time:", end-start)
#     print(callPrice)

# print("M =", 50)
# start = time.time()
# callPrice = dpEuropeanOption(
#     s0, T, K, N=50, r=R, sig=sig, tabulate=False, table=table)
# end = time.time()
# print("time:", end-start)
# print(callPrice)

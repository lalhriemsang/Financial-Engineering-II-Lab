import numpy as np
import matplotlib.pyplot as plt

s0 = 100
K = 105
T = 1
r = 0.05
sig = 0.4


def option(optn="call", M=[1, 5, 10, 20, 50, 100, 200, 400], printPrices=False, tabulate=False, plot=True):

    optionPrice = []

    optn = optn.lower()

    for N in M:
        delt = T/N
        u = np.exp(sig*delt**0.5 + (r-0.5*sig**2)*delt)
        d = np.exp(-sig*delt**0.5 + (r-0.5*sig**2)*delt)
        p0 = (np.exp(r*delt) - d)/(u-d)
        q0 = 1-p0

        if d >= np.exp(r*delt) or np.exp(r*delt) >= u:
            print("Arbitrage Possible!")
            return

        stock = np.zeros((N+1, N+1))
        payOffs = np.zeros((N+1, N+1))
        stock[0][0] = s0

        for i in range(N):
            for j in range(N):
                stock[i][j+1] = u*stock[i][j]
                stock[i+1][j] = d*stock[i][j]

        n = N-1
        i = 0
        j = N-1
        if optn == "call":
            while n >= 0:
                if n == N-1:
                    payOffs[i][j] = (max(stock[i][j+1]-K, 0) +
                                     max(stock[i+1][j]-K, 0))
                else:
                    payOffs[i][j] = (p0*payOffs[i][j+1] +
                                     q0*payOffs[i+1][j])*np.exp(-r*delt)

                i = i+1
                j = j-1

                if j == -1:
                    n -= 1
                    j = n
                    i = 0
        elif optn == "put":
            while n >= 0:
                if n == N-1:
                    payOffs[i][j] = (p0*max(K-stock[i][j+1], 0) +
                                     q0*max(K-stock[i+1][j], 0))*np.exp(-r*delt)
                else:
                    payOffs[i][j] = (p0*payOffs[i][j+1] +
                                     q0*payOffs[i+1][j])*np.exp(-r*delt)

                i = i+1
                j = j-1

                if j == -1:
                    n -= 1
                    j = n
                    i = 0

        optionPrice.append(payOffs[0][0])

        if (N == 20 and tabulate):
            tabulateAtTime = [0, 0.5, 1, 1.5, 2, 3, 4.5]
            tabulatePrices = [[], [], [], [], [], [], []]
            i = 0
            j = 0

            for index in range(len(tabulateAtTime)):
                t = tabulateAtTime[index]
                for i in range(N+1):
                    for j in range(N+1):
                        if (i+j == int(t/delt)):
                            tabulatePrices[index].append(payOffs[i][j])

            print(tabulatePrices)

    if (printPrices):
        print(optionPrice)
    if plot == False:
        return
    if (optn == "call"):
        plt.title("Call pricing")
    elif (optn == "put"):
        plt.title("Put pricing")

    plt.xlabel("time")
    plt.ylabel("price")
    plt.scatter(M, optionPrice, s=5)
    plt.plot(M, optionPrice)
    plt.show()


# 1a
print("Call option:")
option("call", M=[100], printPrices=True, plot=False)
print("Put option:")
option("put", M=[100], printPrices=True, plot=False)

# 1b
# M = range(1, 400, 5)
# option("call", M)
# option("put", M)

# M = range(1, 400, 1)
# option("call", M)
# option("put", M)

# 1c
# print("Tabulting prices for Call:")
# option("call", M=[20], tabulate=True, plot=False)
# print("Tabulting prices for Put:")
# option("put", M=[20], tabulate=True, plot=False)


# option("call", M=[1000], printPrices=True, plot=False)
# option("call", M=[5000], printPrices=True, plot=False)

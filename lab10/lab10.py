import numpy as np
import matplotlib.pyplot as plt


def BGM(s0, mu, sigma, n):

    s = np.zeros(n)
    w = np.random.normal(0, 1, n)
    delt = 1/252
    si_1 = s0

    for i in range(n):
        s[i] = si_1*np.exp((mu-0.5*sigma**2)*delt + sigma*np.sqrt(delt)*w[i])
        si_1 = s[i]

    return s


def q1a():
    s0 = 100
    mu = 0.1
    sigma = 0.2
    r = 0.05
    n = 252
    t = np.arange(n)

    for i in range(10):
        s = BGM(s0, mu, sigma, n)
        plt.plot(t, s)

    plt.title("Real World prices")
    plt.xlabel("t (days)")
    plt.ylabel("S(t)")
    plt.show()

    mu = 0.05
    for i in range(10):
        s = BGM(s0, mu, sigma, n)
        plt.plot(t, s)

    plt.title("Risk Neutral World prices")
    plt.xlabel("t (days)")
    plt.ylabel("S(t)")
    plt.show()


def AsianOption(s0, K, mu, sigma, r, call=True):

    delt = 1/252
    n = 126
    callPayoff = []
    putPayoff = []

    for i in range(1000):
        s = BGM(s0, mu, sigma, n)

        if call == True:
            vT = max(np.mean(s) - K, 0)
            callPayoff.append(vT*np.exp(-r*n*delt))
        else:
            vT = max(K - np.mean(s), 0)
            putPayoff.append(vT*np.exp(-r*n*delt))

    if call == True:
        return np.mean(callPayoff)
    else:
        return np.mean(putPayoff)


def q1b(option="call"):

    option = option.lower()

    s0 = 100
    mu = 0.05
    sigma = 0.2
    r = 0.05

    print(option)

    for K in [90, 105, 110]:
        if option == "call":
            V0 = AsianOption(s0, K, mu, sigma, r)
        else:
            V0 = AsianOption(s0, K, mu, sigma, r, False)
        print(f"K = {K}: s0 = {V0}")


def v0VsS0(option="call"):

    option = option.lower()
    s = np.arange(80, 141)
    v = np.zeros(len(s))

    for i in range(len(s)):

        s0 = s[i]
        mu = 0.05
        sigma = 0.2
        r = mu
        K = 105

        if option == "call":
            V0 = AsianOption(s0, K, mu, sigma, r)
        else:
            V0 = AsianOption(s0, K, mu, sigma, r, False)

        v[i] = V0

    plt.plot(s, v)
    plt.title(f"s0 vs v0 ({option})")
    plt.xlabel("s0")
    plt.ylabel("v0")
    plt.show()


def v0VsK(option="call"):

    option = option.lower()
    k = np.arange(80, 141)
    v = np.zeros(len(k))

    for i in range(len(k)):

        s0 = 100
        mu = 0.05
        sigma = 0.2
        r = mu
        K = k[i]

        if option == "call":
            V0 = AsianOption(s0, K, mu, sigma, r)
        else:
            V0 = AsianOption(s0, K, mu, sigma, r, False)

        v[i] = V0

    plt.plot(k, v)
    plt.title(f"k vs v0 ({option})")
    plt.xlabel("k")
    plt.ylabel("v0")
    plt.show()


def v0VsR(option="call"):

    option = option.lower()
    R = np.linspace(0, 0.5, 150)
    v = np.zeros(len(R))

    for i in range(len(R)):

        s0 = 100
        mu = R[i]
        sigma = 0.2
        K = 110
        r = mu

        if option == "call":
            V0 = AsianOption(s0, K, mu, sigma, r)
        else:
            V0 = AsianOption(s0, K, mu, sigma, r, False)

        v[i] = V0

    plt.plot(R, v)
    plt.title(f"r vs v0 ({option})")
    plt.xlabel("r")
    plt.ylabel("v0")
    plt.show()


def v0VsSigma(option="call"):

    option = option.lower()
    sigmas = np.linspace(0, 1, 120)
    v = np.zeros(len(sigmas))

    for i in range(len(sigmas)):

        s0 = 100
        mu = 0.05
        sigma = sigmas[i]
        K = 110
        r = mu

        if option == "call":
            V0 = AsianOption(s0, K, mu, sigma, r)
        else:
            V0 = AsianOption(s0, K, mu, sigma, r, False)

        v[i] = V0

    plt.plot(sigmas, v)
    plt.title(f"sigma vs v0 ({option})")
    plt.xlabel("sigma")
    plt.ylabel("v0")
    plt.show()


def varianceReduction(x, y):
    muX = np.mean(x)
    muY = np.mean(y)

    num, deno = 0, 0

    for i in range(len(x)):
        num += (x[i] - muX)*(y[i] - muY)
        deno += (x[i] - muX)*(x[i] - muX)

    b = num/deno

    y_ht = np.zeros(len(x))
    for i in range(len(x)):
        y_ht[i] = y[i] - b*(x[i] - muX)

    return y_ht


def reduced_varianceAsianOption(s0, K, mu, sigma, r, call=True):

    delt = 1/252
    n = 126
    callPayoff = []
    putPayoff = []
    control_variate = []

    for i in range(1000):
        s = BGM(s0, mu, sigma, n)

        if call == True:
            vT = max(np.mean(s) - K, 0)
            callPayoff.append(vT*np.exp(-r*n*delt))

            control_variate.append((s[n-1] - K)*np.exp(-r*n*delt))
        else:
            vT = max(K - np.mean(s), 0)
            putPayoff.append(vT*np.exp(-r*n*delt))

            control_variate.append((K - s[n-1])*np.exp(-r*n*delt))

    print("\t  Without Variance reduction \t With Variance Reduction")
    if call == True:
        v = varianceReduction(control_variate, callPayoff)

        print(f"v0: \t  {np.mean(callPayoff)} \t\t {np.mean(v)}")
        print(
            f"volatilty: {np.sqrt(np.var(callPayoff))} \t\t {np.sqrt(np.var(v))}")
    else:
        v = varianceReduction(control_variate, putPayoff)

        print(f"v0: \t  {np.mean(putPayoff)} \t\t {np.mean(v)}")
        print(
            f"volatilty: {np.sqrt(np.var(putPayoff))} \t\t {np.sqrt(np.var(v))}")

    return np.mean(v)


# 1a)
# q1a()

# 1b)
# q1b("call")
# q1b("put")

# 1c)
# v0VsS0("call")
# v0VsS0("put")

# v0VsK("call")
# v0VsK("put")

# v0VsR("call")
# v0VsR("put")

# v0VsSigma("call")
# v0VsSigma("put")

# 2)
s0 = 100
mu = 0.05
sigma = 0.2
r = mu

print("Call")
for K in [90, 105, 110]:
    print(f"K: {K}", end=" ")
    reduced_varianceAsianOption(s0, K, mu, sigma, r)
    print(" ")

print("Put")
for K in [90, 105, 110]:
    print(f"K: {K}", end=" ")
    reduced_varianceAsianOption(s0, K, mu, sigma, r, False)
    print(" ")

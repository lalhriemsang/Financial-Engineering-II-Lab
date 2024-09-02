import numpy as np
import matplotlib.pyplot as plt


def VasicekModel(beta, mu, sigma, r, t, Ts):
    Yield = np.zeros(len(Ts))

    for i, T in enumerate(Ts):
        B = (1 - np.exp(-beta * (T - t))) / beta
        A = np.exp((B - T + t) * (beta * beta * mu - sigma * sigma *
                   0.5) / (beta * beta) - np.power(sigma * B, 2) / (4 * beta))
        P = A * np.exp(-B * r)
        y = -np.log(P) / (T - t)
        Yield[i] = y
    return Yield


def CIRModel(beta, mu, sigma, r, t, Ts):
    Yield = np.zeros(len(Ts))
    gamma = np.sqrt(beta**2 + 2*sigma**2)

    for i, T in enumerate(Ts):
        B = 2*(np.exp(gamma*(T - t)) - 1) / (2*gamma +
                                             (beta+gamma)*(np.exp(gamma*(T - t))-1))
        A = (2*gamma*np.exp((beta+gamma)*(T-t)/2) / (2*gamma +
                                                     (beta+gamma)*(np.exp(gamma*(T - t))-1)))**(2*beta*mu/(sigma**2))
        P = A * np.exp(-B * r)
        y = -np.log(P) / (T - t)
        Yield[i] = y
    return Yield


def q1():

    paramSets = [[5.9, 0.2, 0.3, 0.1], [3.9, 0.1, 0.3, 0.2],
                 [0.1, 0.4, 0.11, 0.1]]

    Ts = np.linspace(0.01, 10, num=10, endpoint=False)
    for i in range(len(paramSets)):

        beta = paramSets[i][0]
        mu = paramSets[i][1]
        sigma = paramSets[i][2]
        r0 = paramSets[i][3]
        Y = VasicekModel(beta, mu, sigma, r0, 0, Ts)
        plt.plot(Ts, Y, marker='o')
        plt.xlabel("T (Maturity)")
        plt.ylabel("Yield")
        plt.title(f"Term structures for 10 unit time set-{i+1}")
        plt.show()

    Ts = np.linspace(1, 500, num=500, endpoint=False)
    for i in range(len(paramSets)):

        for r0 in np.linspace(0.1, 1, 10):

            beta = paramSets[i][0]
            mu = paramSets[i][1]
            sigma = paramSets[i][2]
            Y = VasicekModel(beta, mu, sigma, r0, 0, Ts)
            plt.plot(Ts, Y)
            plt.xlabel("T (Maturity)")
            plt.ylabel("Yield")

        plt.title(f"Term structures for 500 unit time Set-{i+1}")
        plt.show()


def q2():

    paramSets = [[0.02, 0.7, 0.02, 0.1], [0.7, 0.1, 0.3, 0.2],
                 [0.06, 0.09, 0.5, 0.02]]

    Ts = np.linspace(0.1, 10, num=10, endpoint=False)
    for i in range(len(paramSets)):

        beta = paramSets[i][0]
        mu = paramSets[i][1]
        sigma = paramSets[i][2]
        r0 = paramSets[i][3]
        Y = CIRModel(beta, mu, sigma, r0, 0, Ts)
        plt.plot(Ts, Y, marker='o')
        plt.xlabel("T (Maturity)")
        plt.ylabel("Yield")
        plt.title(f"Term structures for 10 unit time set-{i+1}")
        plt.show()

    Ts = np.linspace(1, 600, num=600, endpoint=False)
    print(Ts[len(Ts)-1])
    for i in range(1):
        for r0 in np.linspace(0.1, 1, 10):

            beta = paramSets[i][0]
            mu = paramSets[i][1]
            sigma = paramSets[i][2]
            Y = CIRModel(beta, mu, sigma, r0, 0, Ts)
            plt.plot(Ts, Y)
            plt.xlabel("T (Maturity)")
            plt.ylabel("Yield")

        plt.title(f"Term structures for 600 unit time Set-{i+1}")
        plt.show()


# q1()
q2()

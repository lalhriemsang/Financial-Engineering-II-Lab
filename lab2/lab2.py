import numpy as np
import matplotlib.pyplot as plt

s0 = 100
K = 100
T = 1
M = 100
r = 0.08
sig = 0.3


def binModel(s0, K, T, N, r, sig, Set=1):

    delt = T/N

    if Set == 1:
        u = np.exp(sig*delt**0.5)
        d = np.exp(-sig*delt**0.5)
    elif Set == 2:
        u = np.exp(sig*delt**0.5 + (r-0.5*sig**2)*delt)
        d = np.exp(-sig*delt**0.5 + (r-0.5*sig**2)*delt)

    if d >= np.exp(r*delt) or np.exp(r*delt) >= u:
        print("Arbitrage Possible!")
        return

    p0 = (np.exp(r*delt) - d)/(u-d)
    q0 = 1-p0
    stock = np.zeros((N+1, N+1))
    callPayOffs = np.zeros((N+1, N+1))
    putPayOffs = np.zeros((N+1, N+1))
    stock[0][0] = s0

    for i in range(N):
        for j in range(N):
            stock[i][j+1] = u*stock[i][j]
            stock[i+1][j] = d*stock[i][j]

    n = N-1
    i = 0
    j = N-1
    while n >= 0:
        if n == N-1:
            callPayOffs[i][j] = (p0*max(stock[i][j+1]-K, 0) +
                                 q0*max(stock[i+1][j]-K, 0))*np.exp(-r*delt)

            putPayOffs[i][j] = (p0*max(K-stock[i][j+1], 0) +
                                q0*max(K-stock[i+1][j], 0))*np.exp(-r*delt)
        else:
            callPayOffs[i][j] = (p0*callPayOffs[i][j+1] +
                                 q0*callPayOffs[i+1][j])*np.exp(-r*delt)

            putPayOffs[i][j] = (p0*putPayOffs[i][j+1] +
                                q0*putPayOffs[i+1][j])*np.exp(-r*delt)
        i = i+1
        j = j-1
        if j == -1:
            n -= 1
            j = n
            i = 0

    return [callPayOffs[0][0], putPayOffs[0][0]]


# print("Set-1")
# options = binModel(s0, K, T, M, r, sig, Set=1)
# print("Call price:", options[0])
# print("Put price:", options[1])

# print("Set-2")
# options = binModel(s0, K, T, M, r, sig, Set=2)
# print("Call price:", options[0])
# print("Put price:", options[1])

# # Sensitvity Analysis
fig, axs = plt.subplots(1, 2, figsize=(10, 5))


def q1a():
    s = np.arange(80, 180, 1)

    # Set-1
    call = []
    put = []
    for s0 in s:
        options = binModel(s0, K, T, M, r, sig, Set=1)
        call.append(options[0])
        put.append(options[1])

    axs[0].plot(s, call, color='blue', label='Call', linewidth=1)
    axs[0].plot(s, put, color='red', label='Put', linewidth=1)
    axs[0].set_xlabel('S(0)')
    axs[0].set_ylabel('Option Price')
    axs[0].set_title('Set-1')
    axs[0].legend()
    call = []
    put = []
    for s0 in s:
        options = binModel(s0, K, T, M, r, sig, Set=2)
        call.append(options[0])
        put.append(options[1])

    axs[1].plot(s, call, color='blue', label='Call', linewidth=1)
    axs[1].plot(s, put, color='red', label='Put', linewidth=1)
    axs[1].set_xlabel('S(0)')
    axs[1].set_ylabel('Option Price')
    axs[1].set_title('Set-2')
    axs[1].legend()
    plt.show()


def q1b():
    k = np.arange(95, 200, 1)
    # Set-1
    call = []
    put = []
    for K in k:
        options = binModel(s0, K, T, M, r, sig, Set=1)
        call.append(options[0])
        put.append(options[1])

    axs[0].plot(k, call, color='blue', label='Call', linewidth=1)
    axs[0].plot(k, put, color='red', label='Put', linewidth=1)
    axs[0].set_xlabel('K')
    axs[0].set_ylabel('Option Price')
    axs[0].set_title('Set-1')
    axs[0].legend()
    # Set-2
    call = []
    put = []
    for K in k:
        options = binModel(s0, K, T, M, r, sig, Set=2)
        call.append(options[0])
        put.append(options[1])

    axs[1].plot(k, call, color='blue', label='Call', linewidth=1)
    axs[1].plot(k, put, color='red', label='Put', linewidth=1)
    axs[1].set_xlabel('K')
    axs[1].set_ylabel('Option Price')
    axs[1].set_title('Set-2')
    axs[1].legend()
    plt.show()


def q1c():
    R = np.arange(0.01, 0.3, 0.001)
    # Set-1
    call = []
    put = []
    for r in R:
        options = binModel(s0, K, T, M, r, sig, Set=1)
        call.append(options[0])
        put.append(options[1])

    axs[0].plot(R, call, color='blue', label='Call', linewidth=1)
    axs[0].plot(R, put, color='red', label='Put', linewidth=1)
    axs[0].set_xlabel('r')
    axs[0].set_ylabel('Option Price')
    axs[0].set_title('Set-1')
    axs[0].legend()
    # Set-2
    call = []
    put = []
    for r in R:
        options = binModel(s0, K, T, M, r, sig, Set=2)
        call.append(options[0])
        put.append(options[1])

    axs[1].plot(R, call, color='blue', label='Call', linewidth=1)
    axs[1].plot(R, put, color='red', label='Put', linewidth=1)
    axs[1].set_xlabel('r')
    axs[1].set_ylabel('Option Price')
    axs[1].set_title('Set-2')
    axs[1].legend()
    plt.show()


def q1d():
    Sig = np.arange(0.1, 1, 0.01)
    # Set-1
    call = []
    put = []
    for sig in Sig:
        options = binModel(s0, K, T, M, r, sig, Set=1)
        call.append(options[0])
        put.append(options[1])

    axs[0].plot(Sig, call, color='blue', label='Call', linewidth=1)
    axs[0].plot(Sig, put, color='red', label='Put', linewidth=1)
    axs[0].set_xlabel('sig')
    axs[0].set_ylabel('Option Price')
    axs[0].set_title('Set-1')
    axs[0].legend()
    # Set-2
    call = []
    put = []
    for sig in Sig:
        options = binModel(s0, K, T, M, r, sig, Set=2)
        call.append(options[0])
        put.append(options[1])

    axs[1].plot(Sig, call, color='blue', label='Call', linewidth=1)
    axs[1].plot(Sig, put, color='red', label='Put', linewidth=1)
    axs[1].set_xlabel('sig')
    axs[1].set_ylabel('Option Price')
    axs[1].set_title('Set-2')
    axs[1].legend()
    plt.show()


def q1e():
    m = np.arange(50, 150, 1)
    # Set-1
    call = []
    put = []
    for M in m:
        options = binModel(s0, K, T, M, r, sig, Set=1)
        call.append(options[0])
        put.append(options[1])

    axs[0].plot(m, call, color='blue', label='Call', linewidth=1)
    axs[0].plot(m, put, color='red', label='Put', linewidth=1)
    axs[0].set_xlabel('M')
    axs[0].set_ylabel('Option Price')
    axs[0].set_title('Set-1')
    axs[0].legend()
    # Set-2
    call = []
    put = []
    for M in m:
        options = binModel(s0, K, T, M, r, sig, Set=2)
        call.append(options[0])
        put.append(options[1])

    axs[1].plot(m, call, color='blue', label='Call', linewidth=1)
    axs[1].plot(m, put, color='red', label='Put', linewidth=1)
    axs[1].set_xlabel('M')
    axs[1].set_ylabel('Option Price')
    axs[1].set_title('Set-2')
    axs[1].legend()
    plt.show()

# UNCOMMENT TO RUN

# q1a()
# q1b()
# q1c()
# q1d()
# q1e()

# 2)

# Asian Options


def binModel2(s0, K, T, N, r, sig, Set=1):

    delt = T/N

    if Set == 1:
        u = np.exp(sig*delt**0.5)
        d = np.exp(-sig*delt**0.5)
    elif Set == 2:
        u = np.exp(sig*delt*0.5 + (r-0.5*sig*2)*delt)
        d = np.exp(-sig*delt*0.5 + (r-0.5*sig*2)*delt)

    if d >= np.exp(r*delt) or np.exp(r*delt) >= u:
        print("Arbitrage Possible!")
        return

    p0 = (np.exp(r*delt) - d)/(u-d)
    q0 = 1-p0

    stock = np.zeros(2**(N+1)-1)
    stock[0] = s0
    sum = np.zeros(2**(N+1)-1)
    callPayOffs = np.zeros(2**(N+1)-1)
    putPayOffs = np.zeros(2**(N+1)-1)

    for i in range(N+1):
        stock[2*i+1] = u*stock[i]
        stock[2*i+2] = d*stock[i]

        sum[2*i+1] = sum[i] + stock[2*i+1]
        sum[2*i+2] = sum[i] + stock[2*i+2]

    for i in range(2**N-1, len(sum)):
        callPayOffs[i] = max(sum[i]/N - K, 0)
        putPayOffs[i] = max(K-sum[i]/N, 0)

    i = 2**N-2

    while i >= 0:
        callPayOffs[i] = (p0*callPayOffs[2*i+1] + q0 *
                          callPayOffs[2*i+2])*np.exp(-r*delt)
        putPayOffs[i] = (p0*putPayOffs[2*i+1] + q0 *
                         putPayOffs[2*i+2])*np.exp(-r*delt)
        i = i-1

    return [callPayOffs[0], putPayOffs[0]]


def q2a():
    s = range(50, 150, 5)
    # Set-1
    call = []
    put = []
    for s0 in s:
        options = binModel2(s0, K, T, M, r, sig, Set=1)
        call.append(options[0])
        put.append(options[1])

    plt.scatter(s, call, color='red')
    plt.scatter(s, put, color='red')
    plt.plot(s, call, color='blue', label='Call', linewidth=1)
    plt.plot(s, put, color='red', label='Put', linewidth=1)
    plt.xlabel('S(0)')
    plt.ylabel('Option Price')
    plt.title('Set-1')
    plt.legend()
    plt.show()


def q2b():
    k = range(90, 150, 5)
    # Set-1
    call = []
    put = []
    for K in k:
        options = binModel2(s0, K, T, M, r, sig, Set=1)
        call.append(options[0])
        put.append(options[1])
    plt.scatter(k, call, color='red')
    plt.scatter(k, put, color='red')
    plt.plot(k, call, color='blue', label='Call', linewidth=1)
    plt.plot(k, put, color='red', label='Put', linewidth=1)
    plt.xlabel('K')
    plt.ylabel('Option Price')
    plt.title('Set-1')
    plt.legend()
    plt.show()


def q2c():
    R = np.arange(0.01, 0.09, 0.005)
    # Set-1
    call = []
    put = []
    for r in R:
        options = binModel2(s0, K, T, M, r, sig, Set=1)
        call.append(options[0])
        put.append(options[1])

    plt.scatter(R, call, color='red')
    plt.scatter(R, put, color='red')
    plt.plot(R, call, color='blue', label='Call', linewidth=1)
    plt.plot(R, put, color='red', label='Put', linewidth=1)
    plt.xlabel('r')
    plt.ylabel('Option Price')
    plt.title('Set-1')
    plt.legend()
    plt.show()


def q2d():
    Sig = np.arange(0.1, 1, 0.05)
    # Set-1
    call = []
    put = []
    for sig in Sig:
        options = binModel2(s0, K, T, M, r, sig, Set=1)
        call.append(options[0])
        put.append(options[1])

    plt.scatter(Sig, call, color='red')
    plt.scatter(Sig, put, color='red')
    plt.plot(Sig, call, color='blue', label='Call', linewidth=1)
    plt.plot(Sig, put, color='red', label='Put', linewidth=1)
    plt.xlabel('sig')
    plt.ylabel('Option Price')
    plt.title('Set-1')
    plt.legend()
    plt.show()


def q2e():
    m = np.arange(2, 17, 1)
    # Set-1
    call = []
    put = []
    for M in m:
        options = binModel2(s0, K, T, M, r, sig, Set=1)
        call.append(options[0])
        put.append(options[1])

    plt.scatter(m, call, color='red')
    plt.scatter(m, put, color='red')
    plt.plot(m, call, color='blue', label='Call', linewidth=1)
    plt.plot(m, put, color='red', label='Put', linewidth=1)
    plt.xlabel('M')
    plt.ylabel('Option Price')
    plt.title('Set-1')
    plt.legend()
    plt.show()


plt.subplot(1, 1, 1)

s0 = 100
K = 105
T = 1
M = 10
r = 0.08
sig = 0.3

# UNCOMMENT TO RUN

# q2a()
# q2b()
# q2c()
# q2d()
# q2e()

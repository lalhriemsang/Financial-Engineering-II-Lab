import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import random
from datetime import datetime

df = pd.read_csv("./NIFTYoptiondata.csv")
df = df.iloc[:, 2:]


def q1ai(option="call"):
    option = option.lower()

    K = df['Strike Price  ']
    days = np.zeros(len(df['Expiry  ']))
    if option == "call":
        V = df['Call Price']
    else:
        V = df['Put Price']

    startDate = pd.to_datetime(df['Date  '][0])

    for i in range(len(df['Expiry  '])):
        date = pd.to_datetime(df['Expiry  '][i])
        days[i] = (date - startDate).days

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(K, days, V, s=1.1)
    ax.set_xlabel("Strike Prices")
    ax.set_ylabel("Expiry (days)")
    if option == "call":
        ax.set_zlabel("Call Price")
    else:
        ax.set_zlabel("Put Price")
    plt.show()


def q1aii(option="call"):
    option = option.lower()

    K = df['Strike Price  ']
    days = np.zeros(len(df['Expiry  ']))
    if option == "call":
        V = df['Call Price']
    else:
        V = df['Put Price']

    startDate = pd.to_datetime(df['Date  '][0])

    for i in range(len(df['Expiry  '])):
        date = pd.to_datetime(df['Expiry  '][i])
        days[i] = (date - startDate).days

    fig, axs = plt.subplots(1, 2, figsize=(5*2, 5))
    axs[0].scatter(K, V, s=1.1)
    axs[0].set_xlabel("Strike Prices")
    if option == "call":
        axs[0].set_ylabel("Call Price")
    else:
        axs[0].set_ylabel("Put Price")

    axs[1].scatter(days, V, s=1.1)
    axs[1].set_xlabel("Expiry (days)")
    if option == "call":
        axs[1].set_ylabel("Call Price")
    else:
        axs[1].set_ylabel("Put Price")
    plt.show()


q1aii("call")

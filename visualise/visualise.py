import pandas as pd
import copy
import matplotlib.pyplot as plt

from market_info.market_info import CROSS
from utility.discovery import check_states


def visualise_mymethod(market_infos):
    df = market_infos[-1].df
    df = pd.DataFrame.copy(df)
    mi = copy.deepcopy(market_infos[-1])
    length = df.shape[0]
    states = []
    for i in range(1, length+1):
        tmp = df.head(i)
        mi.df = tmp
        try:
            state = check_states(mi)
        except:
            state = None
        states.append(state)

    pips = df[CROSS + "CLOSE"].shift(-100) - df[CROSS + "CLOSE"]
    pr = pd.DataFrame({'s':states, 'p':list(pips)})
    pr = pr.dropna()
    b = pr[pr['s'] == "BUY"]

    n, bins, patches = plt.hist(b['p'], 50, density=True, facecolor='g', alpha=0.75)
    print(b['p'].mean())
    plt.xlabel('pips')
    plt.ylabel('Probability')
    plt.title('Histogram of BUY')
    plt.grid(True)
    plt.show()

    b = pr[pr['s'] == "SELL"]
    print(b['p'].mean())

    n, bins, patches = plt.hist(b['p'], 50, density=True, facecolor='g', alpha=0.75)

    plt.xlabel('pips')
    plt.ylabel('Probability')
    plt.title('Histogram of SELL')
    plt.grid(True)
    plt.show()



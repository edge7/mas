from market_info.market_info import CROSS
from utility.discovery import check_states, BARAGO
import copy
import pandas as pd


def optimize(market_infos):
    df = market_infos[-1].df
    df = pd.DataFrame.copy(df)
    mi = copy.deepcopy(market_infos[-1])

    states = []
    df = df.tail(3000)
    length = df.shape[0]
    for i in range(1, length + 1):
        tmp = df.head(i)
        mi.df = tmp
        try:
            state = check_states(mi)
        except:
            state = None
        states.append(state)

    pips = df[CROSS + "CLOSE"].shift(-BARAGO) - df[CROSS + "CLOSE"]
    pr = pd.DataFrame({'s': states, 'p': list(pips)})
    pr = pr.dropna()
    b = pr[pr['s'] == "BUY"]
    try:
        buy_mean = b['p'].ewm(span=3).mean().iloc[-1]
    except:
        buy_mean = 1

    b = pr[pr['s'] == "SELL"]
    try:
        sell_mean = b['p'].ewm(span=3).mean().iloc[-1]
    except:
        sell_mean = -1

    d = {'B': buy_mean, 'S': sell_mean}
    print(d)
    return d

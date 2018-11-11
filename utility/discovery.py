import logging
import datetime
from dateutil.parser import parse
import pandas as pd
from matplotlib import pyplot

from utility.mql4_socket import write_trend

BARAGO = 10000000

TP = 60
SL = 60

logger = logging.getLogger(__name__)

from market_info.market_info import MarketInfo, LONG_TREND, CROSS, get_pips

slopes = []
counter = 0


class Trendline(object):
    def __init__(self, t1, t2, t3, m, q, up_down, p1, p2):
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.m = m
        self.q = q
        self.up_down = up_down
        self.p1 = p1
        self.p2 = p2


def get_info(df, news):
    mi = MarketInfo(df, news)
    mi.search_for_info()
    return mi


def search_for_bullish_and_bearish_candlestick(market_info):
    bearish = {'datetime': str(datetime.datetime.now())}
    bullish = {'datetime': str(datetime.datetime.now())}
    market_info = market_info.candle_info
    # Hammer is bullish
    if market_info.hammer:
        bullish['hammer'] = True
    # Hanging man is bearish
    if market_info.hanging:
        bearish['hanging'] = True

    if market_info.inverted_hammer:
        bullish['inv_hammer'] = True

    if market_info.shooting_start:
        bearish['shooting_star'] = True

    if market_info.bullish_engulfing:
        bullish['bullish_eng'] = True

    # if market_info.bearish_engulfing: 'superata_da_
    #    bearish['bearish_eng'] = True

    if market_info.tweezer_tops:
        bullish['tweezer_tops'] = True

    if market_info.tweezer_bottoms:
        bearish['tweezer_bottoms'] = True

    if market_info.morning_star:
        bullish['morning_star'] = True

    if market_info.evening_star:
        bearish['evening_star'] = True

    if market_info.three_white:
        bullish['3_white'] = True

    if market_info.three_black:
        bearish['3_black'] = True

    if market_info.three_ins_up:
        bullish['3_ins_up'] = True

    if market_info.three_ins_down:
        bearish['3_ins_down'] = True

    if market_info.marubozu is not None:
        if market_info.marubozu == 'white':
            bullish['marubozu'] = True
        else:
            bearish['marubozu'] = True

    if market_info.dragonfly:
        bullish['dragonfly'] = True

    if market_info.gravestone:
        bearish['gravestone'] = True

    return bearish, bullish


def in_pips(param):
    if 'JPY' in CROSS or 'XAU' in CROSS:
        multiply = 100.0
    else:
        multiply = 10000.0
    return param / multiply


history_buy = []
history_sell = []


def get_take_profit(mi):
    avg = mi.df[CROSS + "BODY"].tail(10).abs().mean()
    return get_pips(avg) * 4.5


def get_stop_loss(mi):
    avg = mi.df[CROSS + "BODY"].tail(10).abs().mean()
    return get_pips(avg) * 3.5


cc = 0
xx = []
jr = False

CANDLE = 15

orders = {}


def londra(mi):
    df = pd.DataFrame.copy(mi.df)
    df2 = pd.DataFrame.copy(mi.df)
    last_big = mi.df[CROSS + "CLOSE"].ewm(span=100).mean().iloc[-1]
    fast_avg = mi.df[CROSS + "CLOSE"].ewm(span=25).mean().iloc[-1]
    previous_fast_avg = mi.df[CROSS + "CLOSE"].ewm(span=25).mean().iloc[-2]
    current_price = mi.get_close(1)
    close = mi.df[CROSS + "CLOSE"].iloc[-1]
    previous_close = mi.df[CROSS + "CLOSE"].iloc[-2]
    today = df["TIME"].iloc[-1].day

    m = (mi.df[CROSS + "CLOSE"] - mi.df[CROSS + "CLOSE"].ewm(
        span=25).mean()).tail(20).sum()
    m = in_pips(m)
    # s = (mi.df[CROSS + "CLOSE"] - mi.df[CROSS + "CLOSE"].ewm(span=25).mean()).tail(20).std()
    # s = in_pips(s)

    m_ = (mi.df[CROSS + "CLOSE"] - mi.df[CROSS + "CLOSE"].ewm(
        span=25).mean()).tail(21).head(20).sum()
    m_ = in_pips(m)
    # s_ = (mi.df[CROSS + "CLOSE"] - mi.df[CROSS + "CLOSE"].ewm(span=25).mean()).tail(21).head(20).std()
    # s_ = in_pips(s)
    # print("Mean is: " + str(m))
    # print("Std is: " + str(s))

    if close > fast_avg and previous_close < previous_fast_avg and m > 0:
        return "BUY"
    if close < fast_avg and previous_close > previous_fast_avg and m < 0:
        return "SELL"

    return


dates = {}


def check_break(mi):
    df = mi.df
    WINDOW = 20
    min = df[CROSS + "CLOSE"].rolling(window=WINDOW).min().iloc[-2]
    max = df[CROSS + "CLOSE"].rolling(window=WINDOW).max().iloc[-2]

    if get_pips((max - min)) >= 50:
        return

    last_close = df[CROSS + "CLOSE"].iloc[-1]

    if last_close > max:
        return "SELL"
    if last_close < min:
        return "BUY"


old_der = None


def check_derivative(mi):
    global old_der
    df = mi.df
    WINDOW = 15
    der_prima_2 = df[CROSS + "CLOSE"].rolling(window=WINDOW).mean().iloc[-2] - \
                  df[CROSS + "CLOSE"].rolling(window=WINDOW).mean().iloc[-1]
    der_prima_1 = df[CROSS + "CLOSE"].rolling(window=WINDOW).mean().iloc[-3] - \
                  df[CROSS + "CLOSE"].rolling(window=WINDOW).mean().iloc[-2]

    der_2 = der_prima_2 - der_prima_1
    if old_der is None:
        old_der = der_2
        return
    if der_2 > 0 and old_der < 0:
        old_der = der_2
        return "BUY"
    if der_2 < 0 and old_der > 0:
        old_der = der_2
        return "SELL"
    old_der = der_2


def check_beahv2(mi):
    df = mi.df
    df[CROSS + "BODY_INPIPS"] = df[CROSS + "CLOSE"] - df[CROSS + "OPEN"]
    avg = df[CROSS + "BODY_INPIPS"][-30:-1].abs().mean()
    las = df[CROSS + "BODY_INPIPS"].iloc[-1]

    if abs(las) * 0.4 > avg:
        if las > 0:
            return "BUY"
        return "SELL"


def check_beahv(mi):
    df = mi.df
    WINDOW = 75
    dist_avg = df[CROSS + "CLOSE"] - df[CROSS + "CLOSE"].rolling(
        window=WINDOW).mean()
    last_50 = dist_avg.iloc[-25:].sum()
    previous_50 = dist_avg.iloc[-50:-26].sum()

    m = df[CROSS + "CLOSE"].iloc[-15:-1].max()
    min = df[CROSS + "CLOSE"].iloc[-15:-1].min()
    range_10 = df[CROSS + "CLOSE"].iloc[-15:-1].max() - df[
                                                            CROSS + "CLOSE"].iloc[
                                                        -15:-1].min()

    if get_pips(range_10) < 40:
        last_close = df[CROSS + "CLOSE"].iloc[-1]

        if previous_50 < 0 and last_50 > 0 and last_close > m:
            return "BUY"
        if previous_50 > 0 and last_50 < 0 and last_close < min:
            return "SELL"


def check_london_break(market_infos, orders):
    buy = sell = close = scalp = tp = sl = jr = None
    res = None
    fix = None

    if not orders.empty:
        orders.sort_values(by="TIME", ascending=False, inplace=True)
        try:
            seconds_bars = (market_infos[-1].df["TIME"].iloc[-1] -
                            market_infos[-1].df["TIME"].iloc[
                                -2]).total_seconds()

        except Exception:
            seconds_bars = 1
        if seconds_bars == 0: seconds_bars = 1
        seconds_from_orders = (market_infos[-1].df["TIME"].iloc[-1] - parse(
            orders['TIME'].iloc[-1])).total_seconds()
        bar_ago = int(seconds_from_orders / seconds_bars) + 1
        id = str(orders["ID"].iloc[-1])
        current_price = market_infos[-1].get_close(1)
        lots = str(orders["LOTS"].iloc[-1])
        profit = orders["PROFIT"].iloc[-1]

        open_at = orders["OPEN_AT"].iloc[-1]
        if (bar_ago > 24 and profit == 0) or (bar_ago > 6 and profit < 0) or (
                bar_ago > 6 and profit > 0):
            to_close = round(float(lots) / 1, 0)
            if to_close == 0.0:
                to_close = lots
            close = 'CLOSE,' + id + "," + str(lots)

    global dates
    lots = 1.0
    # res = londra(market_infos[-1])
    if res is None:
        res = check_beahv2(market_infos[-1])
        fix = None
    hour = market_infos[-1].df["TIME"].iloc[-1].hour
    day = market_infos[-1].df["TIME"].iloc[-1].date()
    if (not orders.empty) and fix is None:
        res = None
    if res == "SELL":
        sell = True
        buy = False
        last = "SELL"
        tp = market_infos[-1].get_close(1) - in_pips(TP)
        sl = market_infos[-1].get_close(1) + in_pips(SL)
        dates[day] = 1

    if res == "BUY":
        buy = True
        last = "BUY"
        sell = False
        tp = market_infos[-1].get_close(1) + in_pips(TP)
        sl = market_infos[-1].get_close(1) - in_pips(SL)
        dates[day] = 1
    return buy, sell, close, scalp, tp, sl, lots, jr


def count_pos_neg(l):
    pos = 0
    neg = 0
    for item in l:
        if item > 0:
            pos += 1
        if item < 0:
            neg += 1
    return pos, neg


def response(pos, neg, tot, thre=0.8):
    if pos >= tot * thre:
        return "SALE"
    if neg >= tot * thre:
        return "SCENDE"
    return "MIX"


def sii_serio_strong_trend(df):
    avg_50 = df[CROSS + "CLOSE"].rolling(window=50).mean()
    avg_35 = df[CROSS + "CLOSE"].rolling(window=20).mean()
    avg_100 = df[CROSS + "CLOSE"].rolling(window=100).mean()
    avg_200 = df[CROSS + "CLOSE"].rolling(window=200).mean()
    current_close = df[CROSS + "CLOSE"].iloc[-1]
    prev_current_close = df[CROSS + "CLOSE"].iloc[-2]

    closes = df[CROSS + "CLOSE"]
    WEEK1 = 20
    dist_50_WEEK = (closes - avg_50).iloc[-WEEK1:]
    dist_100_WEEK = (closes - avg_100).iloc[-WEEK1:]
    dist_200_WEEK = (closes - avg_200).iloc[-WEEK1:]

    SHORT = 10
    dist_50_SHORT = (closes - avg_50).iloc[-SHORT:]
    dist_100_SHORT = (closes - avg_100).iloc[-SHORT:]
    dist_200_SHORT = (closes - avg_200).iloc[-SHORT:]

    # Count positive negative per avere idea dell'ultima settimana
    pos_50, neg_50 = count_pos_neg(list(dist_50_WEEK))
    pos_100, neg_100 = count_pos_neg(list(dist_100_WEEK))
    pos_200, neg_200 = count_pos_neg(list(dist_200_WEEK))

    week_50_r = response(pos_50, neg_50, WEEK1, thre=0.8)
    week_100_r = response(pos_100, neg_100, WEEK1, thre=0.9)
    week_200_r = response(pos_200, neg_200, WEEK1, thre=0.95)

    # Trade STRONG TREND
    if week_50_r == "SALE" and week_100_r == "SALE" and week_200_r == "SALE":
        if current_close > avg_35.iloc[-1] and prev_current_close < \
                avg_35.iloc[-2]:
            if current_close > avg_50.iloc[-1] and prev_current_close > \
                    avg_50.iloc[-2]:
                return "BUY", "STRONG"

    if week_50_r == "SCENDE" and week_100_r == "SCENDE" and week_200_r == "SCENDE":
        if current_close < avg_35.iloc[-1] and prev_current_close > \
                avg_35.iloc[-2]:
            if current_close < avg_50.iloc[-1] and prev_current_close < \
                    avg_50.iloc[-2]:
                return "SELL", "STRONG"

    # Count positive negative per avere idea dell'ultime 20 ore
    pos_SHORT_50, neg_SHORT_50 = count_pos_neg(list(dist_50_SHORT))
    pos_SHORT_100, neg_SHORT_100 = count_pos_neg(list(dist_100_SHORT))
    pos_SHORT_200, neg_SHORT_200 = count_pos_neg(list(dist_200_SHORT))

    pos_SHORT_50_r = response(pos_SHORT_50, neg_SHORT_50, SHORT, thre=0.95)
    pos_SHORT_100_r = response(pos_SHORT_100, neg_SHORT_100, SHORT, thre=0.9)
    pos_SHORT_200_r = response(pos_SHORT_200, neg_SHORT_200, SHORT, thre=0.9)

    # Se qui, non c'è un super strong trend
    if week_50_r == "MIX" and week_100_r == "MIX" and week_200_r == "MIX":
        if pos_SHORT_50_r == pos_SHORT_100_r == pos_SHORT_200_r == "SALE":
            if current_close > avg_35.iloc[-1] and prev_current_close < \
                    avg_35.iloc[-2]:
                if current_close > avg_50.iloc[-1] and prev_current_close > \
                        avg_50.iloc[-2]:
                    return "BUY", "MIX"

        if pos_SHORT_50_r == pos_SHORT_100_r == pos_SHORT_200_r == "SCENDE":
            if current_close < avg_35.iloc[-1] and prev_current_close > \
                    avg_35.iloc[-2]:
                if current_close < avg_50.iloc[-1] and prev_current_close < \
                        avg_50.iloc[-2]:
                    return "SELL", "MIX"

    # CERCA SUPER BREAKOUT
    if week_50_r == "SALE" and week_100_r == "SALE" and week_200_r == "SALE":
        if get_pips(-current_close + avg_200.iloc[-1]) > 30:
            return "SELL", "BREAK"

    if week_50_r == "SCENDE" and week_100_r == "SCENDE" and week_200_r == "SCENDE":
        if get_pips(current_close - avg_200.iloc[-1]) > 30:
            return "BUY", "BREAK"

    return None, None


last_open = None


def check_se_chiudere(market_infos, last_open, orders):
    close = None
    df = market_infos[-1].df
    avg_50 = df[CROSS + "CLOSE"].rolling(window=50).mean()
    orders.sort_values(by="TIME", ascending=False, inplace=True)
    try:
        seconds_bars = (market_infos[-1].df["TIME"].iloc[-1] -
                        market_infos[-1].df["TIME"].iloc[-2]).total_seconds()

    except Exception:
        seconds_bars = 1
    if seconds_bars == 0: seconds_bars = 1
    seconds_from_orders = (market_infos[-1].df["TIME"].iloc[-1] - parse(
        orders['TIME'].iloc[-1])).total_seconds()
    bar_ago = int(seconds_from_orders / seconds_bars) + 1
    id = str(orders["ID"].iloc[-1])
    current_price = market_infos[-1].get_close(1)
    lots = str(orders["LOTS"].iloc[-1])
    profit = orders["PROFIT"].iloc[-1]

    pos, come = last_open
    if come == "STRONG":
        if pos == "SELL":
            if get_pips(current_price - avg_50.iloc[-1]) > 25:
                close = 'CLOSE,' + id + "," + str(lots)

        if pos == "BUY":
            if get_pips(- current_price + avg_50.iloc[-1]) > 25:
                close = 'CLOSE,' + id + "," + str(lots)

        if close is None and profit > 0 and bar_ago % 5 == 0:
            lots = float(lots) / 4
            lots = round(lots, 2)
            if lots == 0:
                lots = str(orders["LOTS"].iloc[-1])

            close = 'CLOSE,' + id + "," + str(lots)

    return close


def check_avg_behav(market_infos, orders):
    global dates
    global last_open
    mi = market_infos[-1]
    buy = sell = close = scalp = tp = sl = jr = None
    lots = 1.0

    if orders.empty:
        day = market_infos[-1].df["TIME"].iloc[-1].date()
        res, come = sii_serio_strong_trend(mi.df)
        if come == "STRONG":
            TP = 150
            SL = 50
        if come == "MIX":
            TP = 100
            SL = 50
        if come == "BREAK":
            TP = 100
            SL = 50

        if res == "SELL":
            sell = True
            buy = False
            last_open = "SELL", come
            tp = market_infos[-1].get_close(1) - in_pips(TP)
            sl = market_infos[-1].get_close(1) + in_pips(SL)
            dates[day] = 1
            logger.info(str(last_open))

        if res == "BUY":
            buy = True
            last_open = "BUY", come
            sell = False
            tp = market_infos[-1].get_close(1) + in_pips(TP)
            sl = market_infos[-1].get_close(1) - in_pips(SL)
            dates[day] = 1
            logger.info(str(last_open))

    else:

        close = check_se_chiudere(market_infos, last_open, orders)

    return buy, sell, close, scalp, tp, sl, lots, jr


def check_se_qualcosa_trenda(df):
    df[CROSS + "BODY_INPIPS"] = df[CROSS + "CLOSE"] - df[CROSS + "OPEN"]
    last_15_bodies = df[CROSS + "BODY_INPIPS"].iloc[-20:]
    pos, neg = count_pos_neg(list(last_15_bodies))
    avg_25 = (df[CROSS + "CLOSE"] - df[CROSS + "CLOSE"].rolling(
        window=25).mean()).iloc[-15:]
    pos_a, neg_a = count_pos_neg(list(avg_25))
    if pos * 0.6 > neg and pos_a * 0.6 > neg_a:
        return "BUY"
    if neg * 0.6 > pos and neg_a * 0.6 > pos_a:
        return "SELL"


def is_random(df, window=60):
    from scipy import stats
    import numpy
    import matplotlib.pyplot as plt
    from random import randint
    df[CROSS + "BODY_INPIPS"] = df[CROSS + "CLOSE"] - df[CROSS + "OPEN"]
    r = df[CROSS + "BODY_INPIPS"].iloc[-window:]
    m = r.mean()
    s = r.std()
    r = list(r)
    rand = [numpy.random.uniform(m - s, m + s) for i in range(len(r))]
    res = stats.ttest_ind(r, rand)
    p_value = res[1]
    value = res[0]
    if p_value < 0.05:
        print(res)
        print(window)
        # the histogram of the data
        n, bins, patches = plt.hist(r, 50, density=True, facecolor='g',
                                    alpha=0.75)
        print(m)
        plt.xlabel('Smarts')
        plt.ylabel('Probability')
        plt.title('Histogram of IQ')

        plt.grid(True)
        plt.show()
        if value < 0:
            return "SELL"
        else:
            return "BUY"


def check_daily_shit(market_infos, orders):
    buy = sell = close = scalp = tp = sl = jr = None
    lots = 1.0
    global dates
    df = market_infos[-1].df
    che_ore_sono = df["TIME"].iloc[-1].hour
    df = market_infos[-1].df
    day = market_infos[-1].df["TIME"].iloc[-1].date()
    if not orders.empty:

        try:
            seconds_bars = (market_infos[-1].df["TIME"].iloc[-1] -
                            market_infos[-1].df["TIME"].iloc[
                                -2]).total_seconds()

        except Exception:
            seconds_bars = 1
        if seconds_bars == 0: seconds_bars = 1
        seconds_from_orders = (market_infos[-1].df["TIME"].iloc[-1] - parse(
            orders['TIME'].iloc[-1])).total_seconds()
        bar_ago = int(seconds_from_orders / seconds_bars) + 1
        id = str(orders["ID"].iloc[-1])
        current_price = market_infos[-1].get_close(1)
        lots = str(orders["LOTS"].iloc[-1])
        profit = orders["PROFIT"].iloc[-1]

        if che_ore_sono >= 20:
            close = 'CLOSE,' + id + "," + str(lots)

        if che_ore_sono > 15 and profit == 0:
            close = 'CLOSE,' + id + "," + str(lots)

    if (
            che_ore_sono > 10 and che_ore_sono < 17) and orders.empty and dates.get(
        day, 0) == 0:
        res = check_se_qualcosa_trenda(df)
        TP = 50
        SL = 200
        if res == "SELL":
            sell = True
            buy = False
            dates[day] = 1
            tp = market_infos[-1].get_close(1) - in_pips(TP)
            sl = market_infos[-1].get_close(1) + in_pips(SL)

        if res == "BUY":
            dates[day] = 1
            buy = True
            sell = False
            tp = market_infos[-1].get_close(1) + in_pips(TP)
            sl = market_infos[-1].get_close(1) - in_pips(SL)
            logger.info(str(last_open))

    return buy, sell, close, scalp, tp, sl, lots, jr


def check_distro(df, start=20, end=0):
    df[CROSS + "BODY_INPIPS"] = df[CROSS + "CLOSE"] - df[CROSS + "OPEN"]
    df[CROSS + "BODY_INPIPS"] = df[CROSS + "BODY_INPIPS"].rolling(
        window=5).mean()
    r = df[CROSS + "BODY_INPIPS"].iloc[start:end]
    return r.mean(), r.std()


def chart(df):
    df[CROSS + "BODY_INPIPS"] = df[CROSS + "CLOSE"] - df[CROSS + "OPEN"]
    df["TIME"] = df["TIME"].apply(lambda x: x.hour)
    df['STD'] = df[CROSS + "BODY_INPIPS"].rolling(window=10).std()
    res = df.groupby(['TIME']).mean()
    pyplot.plot(res.index, res[CROSS + "BODY_INPIPS"])
    pyplot.show()
    print(res)


def analyse_with_time(df):
    # questo è l'orario della fine della candela, cioè l'orario di ora
    che_ore_sono = df['TIME'].iloc[-1].hour
    che_giorno_e = df['TIME'].iloc[-1].date()
    price = df[CROSS + 'CLOSE'].iloc[-1]
    res = None
    # Prendo solo quelli di oggi
    df = df[df['TIME'].apply(lambda x: x.date()) == che_giorno_e]
    if che_ore_sono == 11:
        # Vediamo il min e max fra le 5 e le 8
        min_5_8 = df[(df['TIME'].apply(lambda x: x.hour) >= 5) & (
                df['TIME'].apply(lambda x: x.hour) <= 8)][
            CROSS + "CLOSE"].min()
        max_5_8 = df[(df['TIME'].apply(lambda x: x.hour) >= 5) & (
                df['TIME'].apply(lambda x: x.hour) <= 8)][
            CROSS + "CLOSE"].max()
        if price > max_5_8:
            res = "BUY"
        elif price < min_5_8:
            res = "SELL"
        elif price > (max_5_8 - min_5_8) * 0.75 + min_5_8:
            res = "BUY"
        elif price < (max_5_8 - min_5_8) * 0.25 + min_5_8:
            res = "SELL"

    if che_ore_sono == 14:
        # Vediamo il min e max fra le 5 e le 8
        min_10_12 = df[(df['TIME'].apply(lambda x: x.hour) >= 10) & (
                df['TIME'].apply(lambda x: x.hour) <= 12)][
            CROSS + "CLOSE"].min()
        max_10_12 = df[(df['TIME'].apply(lambda x: x.hour) >= 10) & (
                df['TIME'].apply(lambda x: x.hour) <= 12)][
            CROSS + "CLOSE"].max()
        if price > max_10_12:
            res = "BUY"
        elif price < min_10_12:
            res = "SELL"
        elif price > (max_10_12 - min_10_12) * 0.75 + min_10_12:
            res = "BUY"
        elif price < (max_10_12 - min_10_12) * 0.25 + min_10_12:
            res = "SELL"
    if res == "SELL":
        res = "BUY"
    elif res == "BUY":
        res = "SELL"
    return res


def check_random_order(market_infos, orders):
    buy = sell = close = scalp = tp = sl = jr = None
    lots = 1.0
    global dates
    df = market_infos[-1].df
    che_ore_sono = df["TIME"].iloc[-1].hour
    df = market_infos[-1].df
    day = market_infos[-1].df["TIME"].iloc[-1].date()
    if not orders.empty:

        try:
            seconds_bars = (market_infos[-1].df["TIME"].iloc[-1] -
                            market_infos[-1].df["TIME"].iloc[
                                -2]).total_seconds()

        except Exception:
            seconds_bars = 1
        if seconds_bars == 0: seconds_bars = 1
        seconds_from_orders = (market_infos[-1].df["TIME"].iloc[-1] - parse(
            orders['TIME'].iloc[-1])).total_seconds()
        bar_ago = int(seconds_from_orders / seconds_bars) + 1
        id = str(orders["ID"].iloc[-1])
        current_price = market_infos[-1].get_close(1)
        lots = str(orders["LOTS"].iloc[-1])
        profit = orders["PROFIT"].iloc[-1]
        if close is None and profit > 0 and bar_ago % 4 == 0 and bar_ago > 20:
            lots = float(lots) / 4
            lots = round(lots, 2)
            if lots == 0:
                lots = str(orders["LOTS"].iloc[-1])
            close = 'CLOSE,' + id + "," + str(lots)

        if che_ore_sono >= 17:
            close = 'CLOSE,' + id + "," + str(lots)

    if dates.get(day, 0) < 3:
        res = None
        res = analyse_with_time(df)
        TP = 40
        SL = 30
        if res == "SELL":
            sell = True
            buy = False
            dates[day] = dates.get(day, 0) + 1
            tp = market_infos[-1].get_close(1) - in_pips(TP)
            sl = market_infos[-1].get_close(1) + in_pips(SL)

        if res == "BUY":
            dates[day] = 1
            buy = True
            sell = False
            tp = market_infos[-1].get_close(1) + in_pips(TP)
            sl = market_infos[-1].get_close(1) - in_pips(SL)
            logger.info(str(last_open))

    return buy, sell, close, scalp, tp, sl, lots, jr


def find_trend_line(trend_line_string, df):
    start = trend_line_string.split("-")[0]
    end = trend_line_string.split("-")[1]
    start_time = parse(start.split(",")[0])
    end_time = parse(end.split(",")[0])
    open_or_close_start = start.split(",")[1]
    open_or_close_end = end.split(",")[1]

    print(trend_line_string)

    if open_or_close_start == "o":
        tmp = "OPEN"
    else:
        tmp = "CLOSE"

    s = df[df["TIME"] == start_time][CROSS + tmp].iloc[-1]

    if open_or_close_end == "o":
        tmp = "OPEN"
    else:
        tmp = "CLOSE"

    e = df[df["TIME"] == end_time][CROSS + tmp].iloc[-1]

    return ((s, start_time), (e, end_time))


trend_line = []
sup_res = []


def close_to_sup_res(df, sup_res):
    current_price = df[CROSS + "CLOSE"].iloc[-1]
    close_to = False
    for s in sup_res:
        if get_pips(abs(s - current_price)) < 10:
            print("Molto vicina a TrendLine")
            close_to = True
    return close_to


def find_x(start, end, df):
    index_start = df[df["TIME"] == start].index[0]
    index_end = df[df["TIME"] == end].index[0]
    return index_end - index_start


def close_to_trend(df, trend_line):
    current_price = df[CROSS + "CLOSE"].iloc[-1]
    close_to = False
    for trend in trend_line:
        y1 = trend[0][0]
        y2 = trend[1][0]
        x1 = 0
        x2 = find_x(trend[0][1], trend[1][1], df)
        m = (y2 - y1) / (x2 - x1)
        q = y1
        x = find_x(trend[0][1], df["TIME"].iloc[-1], df)
        current_y = m * x + q
        if get_pips(abs(current_y - current_price)) < 10:
            print("Molto vicina a TrendLine")
            close_to = True
    return close_to


def get_m(df, window=10, avg=10):
    avg = 25
    df[CROSS + "CLOSE"] = df[CROSS + "CLOSE"].rolling(window=avg).mean()
    y1 = df[CROSS + "CLOSE"].iloc[-window]
    y2 = df[CROSS + "CLOSE"].iloc[-1]
    x1 = 0
    x2 = window - 1
    m = (y2 - y1) / (x2 - x1)
    q = y1
    current_y = m * x2 + q
    return m, current_y


last_pos = "BUY"


def get_range(df):
    m = df[CROSS + "CLOSE"].rolling(window=100).mean()
    m = m.iloc[-24 * 5:]
    mi = m.min()
    ma = m.max()
    return ma - mi


def mini_trend(market_infos, orders):
    global last_pos
    buy = sell = close = scalp = tp = sl = jr = None
    lots = 1
    df = market_infos[-1].df
    res = None
    if not orders.empty:
        try:
            seconds_bars = (market_infos[-1].df["TIME"].iloc[-1] -
                            market_infos[-1].df["TIME"].iloc[
                                -2]).total_seconds()

        except Exception:
            seconds_bars = 1
        if seconds_bars == 0: seconds_bars = 1
        seconds_from_orders = (market_infos[-1].df["TIME"].iloc[-1] - parse(
            orders['TIME'].iloc[-1])).total_seconds()
        bar_ago = int(seconds_from_orders / seconds_bars) + 1
        id = str(orders["ID"].iloc[-1])
        current_price = market_infos[-1].get_close(1)
        lots = str(orders["LOTS"].iloc[-1])
        profit = orders["PROFIT"].iloc[-1]
        m1, y1 = get_m(df.copy(), window=25, avg=10)
        if last_pos == "SELL" and m1 > 0:
            close = 'CLOSE,' + id + "," + str(lots)
        if last_pos == "BUY" and m1 < 0:
            close = 'CLOSE,' + id + "," + str(lots)

        if close is None and bar_ago % 7 == 0 and bar_ago > 5:
            lots_ = 0.1
            if profit < 0:
                lots_ = 0.05
            lots = min(lots_, float(lots))
            lots = round(lots, 2)
            if lots == 0:
                lots = str(orders["LOTS"].iloc[-1])
            close = 'CLOSE,' + id + "," + str(lots)


    else:

        m1, y1 = get_m(df.copy(), window=20, avg=10)
        m2, y2 = get_m(df.copy(), window=30, avg=10)
        r = get_range(df)

        if m1 > 0 and m2 > 0:
            res = "BUY"

        if m1 < 0 and m2 < 0:
            res = "SELL"

        if (get_pips(r) < 30):
            logger.info("NO TRADE")
            res = None
    if res == "SELL":
        sell = True
        buy = False
        tp = market_infos[-1].get_close(1) - in_pips(TP)
        sl = market_infos[-1].get_close(1) + in_pips(SL)
        last_pos = res

    if res == "BUY":
        buy = True
        sell = False
        tp = market_infos[-1].get_close(1) + in_pips(TP)
        sl = market_infos[-1].get_close(1) - in_pips(SL)
        last_pos = res
    return buy, sell, close, scalp, tp, sl, lots, jr


def che_ha_fatto(df, che_ore_sono, window=10):
    current_price = df[CROSS + "CLOSE"].iloc[-1]
    avg_big = df[CROSS + "CLOSE"].rolling(window=200).mean().iloc[-1]
    if current_price > avg_big:
        current_price = df[CROSS + "LOW"].iloc[-1]
        previous_price = df[CROSS + "LOW"].iloc[-2]
    else:
        current_price = df[CROSS + "HIGH"].iloc[-1]
        previous_price = df[CROSS + "HIGH"].iloc[-2]

    std = df[CROSS + "CLOSE"].rolling(window=25).std().iloc[-1]
    avg_small = df[CROSS + "CLOSE"].rolling(window=25).mean()
    band_down = avg_small - std * 2.1
    band_up = avg_small + std * 2.1
    band_down_now = band_down.iloc[-1]
    band_up_now = band_up.iloc[-1]
    band_down_prima = band_down.iloc[-2]
    band_up_prima = band_up.iloc[-2]
    if get_pips(band_up_now - band_down_now) < 30:
        return
    if current_price > avg_big:
        if current_price > band_down_now and previous_price < band_down_prima:
            return "BUY"
    elif current_price < avg_big:
        if current_price < band_up_now and previous_price > band_up_prima:
            return "SELL"

    return


def doCane(market_infos, orders):
    global last_pos
    buy = sell = close = scalp = tp = sl = jr = None
    lots = 1
    res = None
    df = market_infos[-1].df
    che_ore_sono = df['TIME'].iloc[-1].hour
    if not orders.empty:
        try:
            seconds_bars = (market_infos[-1].df["TIME"].iloc[-1] -
                            market_infos[-1].df["TIME"].iloc[
                                -2]).total_seconds()

        except Exception:
            seconds_bars = 1
        if seconds_bars == 0: seconds_bars = 1
        seconds_from_orders = (market_infos[-1].df["TIME"].iloc[-1] - parse(
            orders['TIME'].iloc[-1])).total_seconds()
        bar_ago = int(seconds_from_orders / seconds_bars) + 1
        id = str(orders["ID"].iloc[-1])
        current_price = market_infos[-1].get_close(1)
        lots = str(orders["LOTS"].iloc[-1])
        profit = orders["PROFIT"].iloc[-1]

        if che_ore_sono == 20 or che_ore_sono == 8 or che_ore_sono == 14:
            if profit < -150:
                close = 'CLOSE,' + id + "," + str(lots)

        if close is None and bar_ago % 3 == 0 and profit != 0:
            lots = float(lots)
            lots = 0.1
            close = 'CLOSE,' + id + "," + str(lots)

        if close is None and profit == 0 and bar_ago > 6:
            close = 'CLOSE,' + id + "," + str(lots)

    if True:
        res = 0
        if che_ore_sono >= 21 or che_ore_sono <= 6 or True:
            res = che_ha_fatto(df, che_ore_sono, window=200)
        if che_ore_sono == 8 and False:
            res = che_ha_fatto(df, che_ore_sono, window=15)
        if che_ore_sono == 14 and False:
            res = che_ha_fatto(df, che_ore_sono, window=15)

    if res == "SELL":
        sell = True
        buy = False
        tp = market_infos[-1].get_close(1) - in_pips(TP)
        sl = market_infos[-1].get_close(1) + in_pips(SL)
        last_pos = res

    if res == "BUY":
        buy = True
        sell = False
        tp = market_infos[-1].get_close(1) + in_pips(TP)
        sl = market_infos[-1].get_close(1) - in_pips(SL)
        last_pos = res

    return buy, sell, close, scalp, tp, sl, lots, jr


def ask_me_what_to_do(market_infos, orders):
    global trend_line
    global sup_res
    buy = sell = close = scalp = tp = sl = jr = None
    lots = 1
    return buy, sell, close, scalp, tp, sl, lots, jr

    lots = 1.0
    df = market_infos[-1].df
    che_ore_sono = df['TIME'].iloc[-1].hour
    if che_ore_sono == 20:
        print("Ciao Amigo, cosa facciamo?")
        print("Sup, Res: " + str(sup_res))
        print("Trend Line: " + str(trend_line))
        res = input("Vuoi aggiungere una sup_res?")

        if res == "no":
            print("OK NO")
        else:
            while True:
                res = input("Inserisci (stop per fermarsi)")
                if res == "stop":
                    break
                sup_res.append(float(res))

        res = input("Vuoi aggiungere una Trend Line?")

        if res == "no":
            print("OK NO")
        else:
            res = None
            while True:
                res = input("Inserisci (stop per fermarsi)")
                if res == "stop":
                    break

                startes, endes = find_trend_line(res, df)

                print("Ho trovato queste: ")
                print(startes)
                print(endes)
                res = input("OK?")
                if res == "no":
                    continue
                trend_line.append((startes, endes))

    res = close_to_sup_res(df, sup_res)

    if res:
        input("Open or close? BUY/SELL/CLOSE")
    if res == "BUY":
        buy = True
    if res == "SELL":
        sell = True

    res = close_to_trend(df, trend_line)
    if res:
        input("Open or close? BUY/SELL/CLOSE")
    if res == "SELL":
        sell = True
        buy = False
        tp = market_infos[-1].get_close(1) - in_pips(TP)
        sl = market_infos[-1].get_close(1) + in_pips(SL)

    if res == "BUY":
        buy = True
        sell = False
        tp = market_infos[-1].get_close(1) + in_pips(TP)
        sl = market_infos[-1].get_close(1) - in_pips(SL)

    return buy, sell, close, scalp, tp, sl, lots, jr


def describe_what_is_going_on(market, start, end):
    import copy

    RANGING_THRES = 30  # Improve
    TREND_THRES = 50  # Improve
    if end is not None:
        market = market.iloc[start:end]
    else:
        market = market.iloc[start:]
    max = market.max()
    min = market.min()
    mean = market.mean()
    last = market.iloc[-1]
    first = market.iloc[0]
    range = abs(max - min)
    ranging = False
    down = False

    res = []
    supp = []

    l = list(market)
    for index, value in enumerate(l):
        if index == 0: continue
        THRE = 150
        try:
            if l[index - 1] <= l[index] and (
                    get_pips(l[index - 1] - l[index + 1]) > THRE or get_pips(
                l[index - 1] - l[index + 2]) > THRE or get_pips(
                l[index - 1] - l[index + 4]) > THRE):
                res.append(l[index - 1])
        except Exception:
            pass
        try:
            if l[index - 1] >= l[index] and (
                    get_pips(l[index + 1] - l[index - 1]) > THRE or get_pips(
                l[index + 2] - l[index - 1]) > THRE or get_pips(
                l[index + 4] - l[index - 1]) > THRE):
                supp.append(l[index - 1])
        except Exception:
            pass

    orig = copy.deepcopy(supp)
    new_supp = copy.deepcopy(supp)

    notmod = False
    while not notmod:
        notmod = True
        for item in supp:
            for jitem in supp:
                if item != jitem and get_pips(abs(item - jitem)) < 15:
                    if item in new_supp:
                        new_supp.remove(item)
                    if jitem in new_supp:
                        new_supp.remove(jitem)
                    m = round((item + jitem) / 2, 5)
                    if m not in new_supp:
                        notmod = False
                        new_supp.append(m)
        supp = new_supp

    orig = copy.deepcopy(res)
    new_res = copy.deepcopy(res)

    notmod = False
    while not notmod:
        notmod = True
        for item in res:
            for jitem in res:
                if item != jitem and get_pips(abs(item - jitem)) < 15:
                    if item in new_res:
                        new_res.remove(item)
                    if jitem in new_res:
                        new_res.remove(jitem)
                    m = round((item + jitem) / 2, 5)
                    if m not in new_res:
                        notmod = False
                        new_res.append(m)
        res = new_res
    up = 1
    return ranging, up, down, res, supp, list(set(res + supp))


def scalp_man(df):
    trigger = tp = sl = action = None
    df[CROSS + "BODY_INPIPS"] = df[CROSS + "CLOSE"] - df[CROSS + "OPEN"]
    bb = df[CROSS + "BODY_INPIPS"]
    big = df[CROSS + "CLOSE"].ewm(span=300).mean().iloc[-1]
    p = df[CROSS + "CLOSE"].iloc[-1]
    p_p = df[CROSS + "CLOSE"].iloc[-2]
    BIG = 50
    SMALL = 5
    big = df[CROSS + "CLOSE"].ewm(span=BIG).mean()
    small = df[CROSS + "CLOSE"].ewm(span=SMALL).mean()
    a_20 = abs(df[CROSS + "CLOSE"].iloc[-30] - p)
    no_abs = p - df[CROSS + "CLOSE"].iloc[-30]
    s_20 = bb.apply(lambda x: abs(x)).rolling(window=30).sum().iloc[-1]

    res = a_20 / s_20
    _, _, _, _, _, sups = describe_what_is_going_on(df[CROSS + "CLOSE"],
                                                    start=1, end=None)

    ok = False
    for i in sups:
        if abs(get_pips(i - p)) < 10:
            ok = True
            break

    if not ok:
        return trigger, tp, sl, action

    if no_abs > 0:
        action = "BUY"
        tp = p + in_pips(70)
        sl = p - in_pips(40)
        trigger = p + in_pips(6)

    if no_abs < 0:
        action = "SELL"
        tp = p - in_pips(70)
        sl = p + in_pips(40)
        trigger = p - in_pips(6)

    return trigger, tp, sl, action


def check_if_most_are_down(m, x2, q, closes):
    how_many_up = 0
    how_many_down = 0
    total = 0
    for index, value in enumerate(closes):
        if index <= x2:
            continue

        total += 1
        x = index - x2
        y = m * x + q
        if y > value:
            how_many_up += 1
        if y < value:
            how_many_down += 1
    return how_many_up, how_many_down, total

def find_optimal_to_buy(df, result = []):
    df_copy = df.copy()

    for i in df_copy.index:
        i = i + 1
        if i <= len(result):
            continue
        tmp = df.iloc[0:i]
        if tmp.empty:
            continue
        try:
            trends = find_trends(tmp)
            res = check_if_buy(trends, tmp, log = False)
        except:
            pass



def find_trends(df):
    trends = []
    df = df.copy()
    d = {}
    for n in [150]:
        last_n_closes = df[CROSS + "CLOSE"].iloc[-n:]
        last_n_times = df["TIME"].iloc[-n:]
        last_n_closes = list(last_n_closes)
        last_n_times = list(last_n_times)
        # Praticamente, devi prendere ogni punto, fare la retta con tutti
        # gli altri e vedere se la retta tocca almeno un altro
        for index, primo in enumerate(last_n_closes):
            AVANTI_1 = 2
            if index > len(last_n_closes) / 3:
                pass
            for index_2, second in enumerate(last_n_closes):
                if index_2 <= index + AVANTI_1:
                    continue
                y2 = second
                y1 = primo
                x2 = index_2 - index
                x1 = 0
                m = (y2 - y1) / (x2 - x1)
                q = (x2 * y1 - x1 * y2) / (x2 - x1)
                AVANTI_2 = 2
                for index_3, terzo in enumerate(
                        last_n_closes):
                    if index_3 <= index_2 + AVANTI_2:
                        continue
                    x = index_3 - index
                    y = m * x + q
                    if abs(get_pips(y - terzo)) < 5.1 and index_3 - index > 15:
                        how_many_up, how_many_down, total = check_if_most_are_down(m, index, q, last_n_closes)
                        if how_many_up / total > 0.98:
                            t1 = last_n_times[index]
                            t2 = last_n_times[index_2]
                            t3 = last_n_times[index_3]
                            tr = Trendline(t1, t2, t3, m, q, "UP", primo, second)
                            hash = str(round(m, 5)) + "_" + str(round(q, 5))
                            if hash not in d:
                                trends.append(tr)
                            d[hash] =1
                        if how_many_down / total > 0.98:
                            t1 = last_n_times[index]
                            t2 = last_n_times[index_2]
                            t3 = last_n_times[index_3]
                            tr = Trendline(t1, t2, t3, m, q, "DOWN", primo, second)
                            hash = str(round(m, 5)) + "_" + str(round(q, 5))
                            if hash not in d:
                                trends.append(tr)
                            d[hash] = 1

    return trends


def search_buy_trend(trends, df, no_risk=True):
    p = df[CROSS + "CLOSE"].iloc[-1]
    for t in trends:
        if t.up_down == "UP" and no_risk:
            continue
        m = t.m
        q = t.q
        tim = t.t1
        old_index = list(df[df["TIME"] == tim].index)
        assert len(old_index) == 1
        old_index = old_index[0]
        x = list(df.index)[-1] - old_index
        y = m * x + q
        if 0 < (get_pips(p - y)) < 10 and m > 0:
            return "BUY", t
    return None, None


def search_sell_trend(trends, df, no_risk=True):
    p = df[CROSS + "CLOSE"].iloc[-1]
    for t in trends:
        if t.up_down == "DOWN" and no_risk:
            continue
        m = t.m
        q = t.q
        tim = t.t1
        old_index = list(df[df["TIME"] == tim].index)
        assert len(old_index) == 1
        old_index = old_index[0]
        x = list(df.index)[-1] - old_index
        y = m * x + q
        if 0 < (get_pips(y - p)) < 10 and m < 0:
            return "SELL", t
    return None, None


def check_if_buy(trends, df, log = True):
    avg_100 = df[CROSS + "CLOSE"].rolling(window=100).mean().iloc[-1]
    p = df[CROSS + "CLOSE"].iloc[-1]
    res = None
    limit = False

    if p > avg_100:
        res, t = search_buy_trend(trends, df)
    else:
        res, t = search_sell_trend(trends, df)

    if res == "BUY" or res == "SELL":
        write_trend(t)

    if res is None and False:
        if p > avg_100:
            res, t = search_buy_trend(trends, df, no_risk=False)
        else:
            res, t = search_sell_trend(trends, df, no_risk=False)

        if (res == "BUY" or res == "SELL") and log:
            write_trend(t)
            limit = True

    if res is None:
        if trends and log:
            write_trend(trends[-1])
    return res, limit


YOCLOSE = 15
def figa(df, orders):
    global dates

    trigger = tp = sl = action = close = limit = None
    df = df.copy()
    if not orders.empty:
        while not orders.empty and close is None:

            try:
                seconds_bars = (df["TIME"].iloc[-1] -
                                df["TIME"].iloc[
                                    -2]).total_seconds()

            except Exception:
                seconds_bars = 1

            if seconds_bars == 0: seconds_bars = 1
            seconds_from_orders = (df["TIME"].iloc[-1] - parse(
                orders['TIME'].iloc[-1])).total_seconds()
            bar_ago = int(seconds_from_orders / seconds_bars) + 1
            id = str(orders["ID"].iloc[-1])
            lots = str(orders["LOTS"].iloc[-1])
            profit = orders["PROFIT"].iloc[-1]
            day = df["TIME"].iloc[-1].date()
            dates[day] = 1
            if bar_ago > YOCLOSE:
                close = 'CLOSE,' + id + "," + str(lots)

            if close is None and bar_ago > 3 and bar_ago % 3 == 0 and profit != 0:
                lots_ = lots
                lots = float(lots) / 5.0
                lots = round(lots, 3)
                if lots == 0:
                    lots = lots_
                close = 'CLOSE,' + id + "," + str(lots)

            if bar_ago > 18 and profit < 0:
                lots_ = lots
                lots = float(lots) / 1.0
                lots = round(lots, 3)
                if lots == 0:
                    lots = lots_
                close = 'CLOSE,' + id + "," + str(lots)

            if bar_ago > 10 and profit == 0:
                close = 'CLOSE,' + id + "," + str(lots)

            orders = orders.iloc[:-1]

    else:
            p = df[CROSS + "CLOSE"].iloc[-1]
            trends = find_trends(df)
            action, limit = check_if_buy(trends, df)
            if not limit:
                if action == "SELL":
                    trigger = p - in_pips(50)
                    p = trigger
                    tp = p - in_pips(200)
                    sl = p + in_pips(50)
                if action == "BUY":
                    trigger = p + in_pips(50)
                    p = trigger
                    tp = p + in_pips(200)
                    sl = p - in_pips(50)


    return trigger, tp, sl, action, close

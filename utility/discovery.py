import logging
import datetime
from dateutil.parser import parse
import pandas as pd


BARAGO = 10000000

TP = 60
SL = 25

logger = logging.getLogger(__name__)

from market_info.market_info import MarketInfo, LONG_TREND, CROSS, get_pips

slopes = []
counter = 0


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

    m = (mi.df[CROSS + "CLOSE"] - mi.df[CROSS + "CLOSE"].ewm(span=25).mean()).tail(20).sum()
    m = in_pips(m)
    # s = (mi.df[CROSS + "CLOSE"] - mi.df[CROSS + "CLOSE"].ewm(span=25).mean()).tail(20).std()
    # s = in_pips(s)

    m_ = (mi.df[CROSS + "CLOSE"] - mi.df[CROSS + "CLOSE"].ewm(span=25).mean()).tail(21).head(20).sum()
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


def check_london_break(market_infos, orders):
    buy = sell = close = scalp = tp = sl = jr = None

    if not orders.empty:
        try:
            seconds_bars = (
                    market_infos[-1].df["TIME"].iloc[-1] -
                    market_infos[-1].df["TIME"].iloc[-2]).total_seconds()

        except Exception:
            seconds_bars = 1
        if seconds_bars == 0: seconds_bars = 1
        seconds_from_orders = (
                market_infos[-1].df["TIME"].iloc[-1] - parse(orders['TIME'].iloc[-1])).total_seconds()
        bar_ago = int(seconds_from_orders / seconds_bars) + 1
        id = str(orders["ID"].iloc[-1])
        current_price = market_infos[-1].get_close(1)
        lots = str(orders["LOTS"].iloc[-1])
        profit = orders["PROFIT"].iloc[-1]

        open_at = orders["OPEN_AT"].iloc[-1]
        if (bar_ago > 4 and profit == 0) or (bar_ago > 5 and profit < 0) or (bar_ago > 4 and profit > 0):
            to_close = round(float(lots) / 1, 0)
            if to_close == 0.0:
                to_close = lots
            close = 'CLOSE,' + id + "," + str(lots)
    global dates
    lots = 1.0
    res = londra(market_infos[-1])

    hour = market_infos[-1].df["TIME"].iloc[-1].hour
    day = market_infos[-1].df["TIME"].iloc[-1].date()
    if hour < 8 or hour > 13 or not orders.empty or day in dates:
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

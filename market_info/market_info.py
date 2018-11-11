from dateutil.parser import parse

CROSS = "USDJPY_"
# USDJPY, USDCAD OTTIMI Inizio 2017-05-03
#CADJPY NO, perchè sembra poco liquido per such a TP
from copy import deepcopy

LONG_TREND = CROSS + 'DISTAVG200'
MEDIUM_TREND = CROSS + 'DISTAVG100'
SHORT_TREND = CROSS + 'DISTAVG25'
import logging

HIGH_BAND_50 = CROSS + 'HIGHBAND50'

LOW_BAND_50 = CROSS + 'LOWBAND50'
logger = logging.getLogger(__name__)


def in_pips(param):
    if 'JPY' in CROSS or 'XAU' in CROSS:
        multiply = 100.0
    else:
        multiply = 10000.0
    return param / multiply


def get_pips(body):
    if 'JPY' in CROSS or 'XAU' in CROSS :
        multiply = 100.0
    else:
        multiply = 10000.0
    return body * multiply


class CandleStickInfo(object):
    def __init__(self, df, mi):
        self.df = df
        self.SPINNING_THRES = 3
        self.MAR = 10
        self.spinning_tops = None
        self.mi = mi

    def search_patterns(self):
        return
        self.spinning_tops = self.search_spinning_top()
        self.marubozu = self.search_marubozu()
        self.hammer = self.search_hammer()
        self.hanging = self.search_hanging()
        self.inverted_hammer = self.search_inverted_hammer()
        self.shooting_start = self.search_shooting_star()
        self.bullish_engulfing = self.search_bullish_engulfing()

        self.bearish_engulfing = self.search_bearish_engulfing()


        self.tweezer_bottoms = self.search_tweez_bot()
        self.tweezer_tops = self.search_tweez_top()
        self.evening_star = self.search_evening_star()
        self.morning_star = self.search_moning_star()
        self.three_white = self.search_three_white()
        self.three_black = self.search_three_black()
        self.three_ins_up = self.search_3_up()
        self.three_ins_down = self.search_3_down()
        self.dragonfly = self.search_dragonfly_doji()
        self.gravestone = self.search_gravestone_doji()


    def search_spinning_top(self):
        last_body = abs(self.get_body(1))
        last_high = self.get_high(1)
        last_low = self.get_low(1)
        if last_body * self.SPINNING_THRES < last_high and last_body * self.SPINNING_THRES < last_low \
                and last_high / last_low < 1.8:
            logger.info("Found Spinning TOP")
            return True
        return False

    def get_body(self, i):
        return self.df["BODY"].iloc[-1 * i]

    def get_high(self, i):
        return self.df["HIGHINPIPS"].iloc[-1 * i]

    def get_low(self, i):
        return self.df["LOWINPIPS"].iloc[-1 * i]

    def get_close(self, i):
        return self.df['CLOSE'].iloc[-1 * i]

    def get_open(self, i):
        return self.df['OPEN'].iloc[-1 * i]

    def search_marubozu(self):
        last_body = abs(self.get_body(1))
        if get_pips(abs(last_body)) < 13.0: return None
        last_high = self.get_high(1)
        last_low = self.get_low(1)
        if last_high * self.MAR < last_body and last_body > last_low * self.MAR:
            if self.get_body(1) > 0:
                logger.info("Found White Marubozu")
                return 'white'
            else:
                logger.info("Found Black Marubozu")
                return 'black'
        return None

    def search_hammer(self):
        is_bearish = -1
        if is_bearish != -1:
            return False

        if not (self.get_close(1) < self.get_close(2) < self.get_close(3)) < self.get_close(4) < self.get_close(5):
            return False

        if not self.get_close(1) < self.get_close(6):
            return False

        last_body = abs(self.get_body(1))
        last_high = self.get_high(1)
        last_low = self.get_low(1)
        if last_low > 2.5 * abs(last_body) and last_low > 5 * last_high:
            logger.info("Found Hammer in Downtrend")
            return True

        return False

    def search_hanging(self):
        is_bullish = 1
        if is_bullish != 1:
            return False

        if not (self.get_close(1) > self.get_close(2) > self.get_close(3) > self.get_close(4)):
            return False

        if self.get_close(1) < self.get_close(5):
            return False

        last_body = abs(self.get_body(1))
        last_high = self.get_high(1)
        last_low = self.get_low(1)

        if last_low > 2.5 * abs(last_body) and last_low > 5 * last_high:
            logger.info("Found Hanging in UpTrend")
            return True

        return False

    def search_inverted_hammer(self):
        is_bearish = -1
        if is_bearish != -1:
            return False

        if not (self.get_close(1) < self.get_close(2) < self.get_close(3) < self.get_close(4)):
            return False

        last_body = abs(self.get_body(1))
        last_high = self.get_high(1)
        last_low = self.get_low(1)
        if last_high > 2.5 * abs(last_body) and last_high > 5 * last_low:
            logger.info("Found Inverted Hammer in Downtrend")
            return True

        return False

    def search_shooting_star(self):
        is_bullish = 1
        if is_bullish != 1:
            return False

        if not (self.get_close(1) > self.get_close(2) > self.get_close(3)):
            return False

        if self.get_close(1) < self.get_close(4):
            return False

        last_body = abs(self.get_body(1))
        last_high = self.get_high(1)
        last_low = self.get_low(1)
        if last_high > 2.5 * abs(last_body) and last_high > 5 * last_low:
            logger.info("Found Shooting star in UpTrend")
            return True

        return False

    def search_bullish_engulfing(self):
        is_bearish = True
        if not is_bearish:
            return False
        last_body = self.get_body(1)
        previous_body = self.get_body(2)
        pips_body = get_pips(abs(last_body))
        if pips_body > 12 and last_body > 0 > previous_body and abs(last_body) > abs(previous_body) * 1.5:
            logger.info("Bullish engulfing candle")
            return True

        return False

    def search_bearish_engulfing(self):
        is_bullish = True
        if not is_bullish:
            return False

        last_body = self.get_body(1)
        previous_body = self.get_body(2)
        pips_body = abs(get_pips(last_body))
        if pips_body > 12 and last_body < 0 < previous_body and abs(last_body) > abs(previous_body) * 1.5:
            logger.info("Bearish engulfing candle")
            return True

        return False

    def search_tweez_bot(self):
        is_bearish = -1
        if is_bearish != -1:
            return False
        last_body = self.get_body(1)
        previous_body = self.get_body(2)
        last_low = self.get_low(1)
        previous_low = self.get_low(2)
        if previous_body < 0 < last_body and (1.10 > last_low / previous_low > 0.90) and \
                (1.10 > abs(last_body / previous_body) > 0.90):
            logger.info("Tweez Bot")
            return True
        return False

    def search_tweez_top(self):
        is_bullish = 1
        if is_bullish != 1:
            return False
        last_body = self.get_body(1)
        previous_body = self.get_body(2)
        last_high = self.get_high(1)
        previous_high = self.get_high(2)

        if previous_body > 0 > last_body and (1.10 > last_high / previous_high > 0.90) and \
                (1.10 > abs(last_body / previous_body) > 0.90):
            logger.info("Tweez Tops!")
            return True
        return False

    def search_evening_star(self):
        is_bullish = 1
        if is_bullish != 1:
            return False
        if self.get_close(3) < self.get_close(7):
            return False
        body_3 = self.get_body(3)
        if body_3 < 0: return False
        previous_body = self.get_body(2)
        last_body = self.get_body(1)
        if abs(previous_body) * 3 < abs(body_3) and abs(previous_body) * 3 < abs(last_body) \
                and last_body < 0 and self.get_close(1) < self.get_close(3) - body_3 * 0.55:
            logger.info("Evening Star")
            return True
        return False

    def search_dragonfly_doji(self):
        close = self.get_close(1)
        close_ago = self.get_close(4)
        close_ago_ago = self.get_close(7)
        close_ago_ago_ago = self.get_close(13)
        if close < close_ago and close < close_ago_ago and close < close_ago_ago_ago:
            body = self.get_body(1)
            high = self.get_high(1)
            low = self.get_low(1)
            if abs(body) < in_pips(5) < low and high < in_pips(5) and low * 0.75 > high and get_pips(low) > 10:
                return True
        return False

    def search_gravestone_doji(self):
        close = self.get_close(1)
        close_ago = self.get_close(4)
        close_ago_ago = self.get_close(7)
        close_ago_ago_ago = self.get_close(13)
        if close > close_ago and close > close_ago_ago and close > close_ago_ago_ago:
            body = self.get_body(1)
            high = self.get_high(1)
            low = self.get_low(1)
            if abs(body) < in_pips(5) < high and low < in_pips(5) and high * 0.75 > low and get_pips(high) > 10:
                return True
        return False


    def search_moning_star(self):
        is_bearish = -1
        if is_bearish != -1:
            return False
        body_3 = self.get_body(3)
        if body_3 > 0: return False
        if self.get_close(3) > self.get_close(7) and self.get_close(3) > self.get_close(5):
            return False
        previous_body = self.get_body(2)
        last_body = self.get_body(1)
        if abs(last_body) > self.get_close(1) * 1.2 and abs(previous_body) * 3 < abs(body_3) and abs(
                previous_body) * 3 < abs(last_body) \
                and last_body > 0 and self.get_close(1) > self.get_close(3) - body_3 * 0.55:
            logger.info("Morning Star Bullish signal")
            return True
        return False

    def search_three_white(self):
        is_bearish = True
        if not is_bearish:
            return False
        last_body = self.get_body(1)
        previous_body = self.get_body(2)
        body_3 = self.get_body(3)
        if last_body < 0 or previous_body < 0 or body_3 < 0:
            return False
        if previous_body < body_3:
            return False
        previous_high = self.get_high(2)
        if previous_high * 5 > previous_body:
            return False
        if last_body >= previous_body and last_body / self.get_high(1) > 1.5 and last_body > self.get_low(1):
            logger.info("Got 3 soldiers .. Buy this shit")
            return True
        return False

    def search_three_black(self):
        is_bearish = False
        if is_bearish:
            return False
        last_body = self.get_body(1)
        previous_body = self.get_body(2)
        body_3 = self.get_body(3)
        if last_body > 0 or previous_body > 0 or body_3 > 0:
            return False
        if abs(previous_body) < abs(body_3):
            return False
        previous_high = self.get_high(2)
        if previous_high * 5 > previous_body:
            return False
        if abs(last_body) >= abs(previous_body) and abs(last_body) / self.get_high(1) > 1.5 and abs(
                last_body) > self.get_low(1):
            logger.info("Got 3 soldiers .. Sell this shit")
            return True
        return False

    def search_3_up(self):
        is_bearish = True
        if not is_bearish:
            return False
        if self.get_close(3) < self.get_close(2) < self.get_close(1) \
                and self.get_body(2) > 0 \
                and self.get_body(1) > 0 \
                and self.get_body(3) < 0 and abs(self.get_body(2)) > abs(self.get_body(3)) * 0.6 \
                and self.get_close(1) > self.get_open(3) + self.get_high(3):
            logger.info("3 Inside UP")
            return True

        return False

    def search_3_down(self):
        is_bearish = False
        if is_bearish:
            return False
        if self.get_close(3) > self.get_close(2) > self.get_close(1) \
                and self.get_body(2) < 0 \
                and self.get_body(1) < 0 \
                and self.get_body(3) > 0 and abs(self.get_body(2)) > abs(self.get_body(3)) * 0.6 \
                and self.get_close(1) < self.get_open(3) - self.get_low(3):
            logger.info("3 Inside down")
            return True


class MarketInfo(object):

    def __init__(self, df, news):
        self.news = news
        self.df = df
        self.short_trend = None
        self.long_trend = None
        self.medium_trend = None
        self.low_band_50 = None
        self.high_band_50 = None

    def get_body(self, i):
        return self.df[CROSS + "BODY"].iloc[-1 * i]

    def get_high(self, i):
        return self.df[CROSS + "HIGHINPIPS"].iloc[-1 * i]

    def get_low(self, i):
        return self.df[CROSS + "LOWINPIPS"].iloc[-1 * i]

    def get_close(self, i):
        return self.df[CROSS + 'CLOSE'].iloc[-1 * i]

    def get_open(self, i):
        return self.df[CROSS + 'OPEN'].iloc[-1 * i]

    def search_for_info(self):
        return
        self.read_news()
        return
        self.search_for_short_trend()
        self.search_long_trend()
        self.search_medium_trend()
        self.search_mov_against_short()
        self.search_mov_against_long()
        self.search_high_band_50()
        self.search_low_band_50()
        self.search_regime_switching()
        self.search_dist_short()
        self.candle_info = CandleStickInfo(self.remove(deepcopy(self.df.tail(200))), self)
        self.candle_info.search_patterns()

    def search_dist_short(self):
        l = list(self.df[MEDIUM_TREND])
        l = [abs(value) for value in l]
        l.sort()
        current_value = self.df[MEDIUM_TREND].iloc[-1]
        index = 0
        for v in l:
            if v >= abs(current_value):
                break
            index += 1
        res = index / len(l)
        if res > 0.85:
            logger.info("REVERT!!")

    def search_long_trend(self):
        #  Close - AVG, >0 ==> up_trend
        self.long_trend = self.df.tail(4)[LONG_TREND] > 0

        try:
            up = self.long_trend.groupby(self.long_trend).count()[True]
        except KeyError as e:
            up = 0
        try:
            down = self.long_trend.groupby(self.long_trend).count()[False]
        except KeyError as e:
            down = 0

        if up > 0.65 * (up + down):
            # logger.info("Long Trend is bullish")
            self.long_trend_flag = 1
        elif down > 0.65 * (up + down):
            # logger.info("Long trend is bearish")
            self.long_trend_flag = -1
        else:
            # logger.info("Long trend consolidating")
            self.long_trend_flag = 0

    def search_for_short_trend(self):
        #  Close - AVG, >0 ==> up_trend
        self.short_trend = self.df.tail(10)[SHORT_TREND] > 0
        try:
            up = self.short_trend.groupby(self.short_trend).count()[True]
        except KeyError as e:
            up = 0
        try:
            down = self.short_trend.groupby(self.short_trend).count()[False]
        except KeyError as e:
            down = 0

        if up > 0.65 * (up + down):
            # logger.info("Short Trend is bullish")
            self.short_trend_flag = 1
        elif down > 0.65 * (up + down):
            # logger.info("Short trend is bearish")
            self.short_trend_flag = -1
        else:
            # logger.info("Short trend consolidating")
            self.short_trend_flag = 0

    def search_medium_trend(self):
        self.medium_trend = self.df.tail(4)[MEDIUM_TREND] > 0
        try:
            up = self.medium_trend.groupby(self.medium_trend).count()[True]
        except KeyError as e:
            up = 0
        try:
            down = self.medium_trend.groupby(self.medium_trend).count()[False]
        except KeyError as e:
            down = 0

        if up > 0.65 * (up + down):
            # logger.info("Medium Trend is bullish")
            self.medium_trend_flag = 1
        elif down > 0.65 * (up + down):
            self.medium_trend_flag = -1
            # logger.info("Medium trend is bearish")
        else:
            self.medium_trend_flag = 0
            # logger.info("Medium trend consolidating")

    def search_low_band_50(self):
        self.low_band_50 = self.df.tail(10)[LOW_BAND_50] < 0
        last_value = self.low_band_50.iloc[-1]
        if last_value:
            logger.info("Low band 50")

    def search_high_band_50(self):
        self.high_band_50 = self.df.tail(10)[HIGH_BAND_50] > 0
        last_value = self.high_band_50.iloc[-1]
        if last_value:
            logger.info("High band 50")

    def search_regime_switching(self):
        st_1 = self.short_trend.iloc[-1]
        st_2 = self.short_trend.iloc[-2]
        self.going_up_quickly = False
        self.going_down_quickly = False

        def all_down(l):
            res = False
            for i in range(3, 6):
                res = res or l.iloc[-i]
            return not res

        if (st_1 or st_2) and all_down(self.short_trend):
            # Above short AVG
            if self.get_body(1) > 0 and self.get_body(2) > 0 \
                    and get_pips(self.get_body(1) + self.get_body(2)) > 5 and self.long_trend.iloc[-1]:
                self.going_up_quickly = True
            elif self.get_body(1) < 0 and self.get_body(2) > 0 and not self.long_trend.iloc[-1]:
                self.going_down_quickly = True

        def all_up(l):
            res = True
            for i in range(3, 6):
                res = res and l.iloc[-i]
            return res

        if (not st_1 or not st_2) and all_up(self.short_trend):
            # Belowe short AVG
            if self.get_body(1) < 0 and self.get_body(2) < 0 \
                    and abs(get_pips(self.get_body(1) + self.get_body(2))) > 5 and not self.medium_trend.iloc[-1]:
                self.going_down_quickly = True
            elif self.get_body(1) > 0 and self.medium_trend.iloc[-1]:
                self.going_up_quickly = True

    @staticmethod
    def remove(df):
        for row in list(df.columns):
            df[row.replace(CROSS, '')] = df[row]
            del df[row]
        return df

    def get_response(self, a, b, news):
        def reply(input):
            return float(
                input.replace("K", '1000').replace("M", '1000000').replace('%', '').replace("B", "1000000000").replace(
                    "T", "10000000").replace(",", "."))

        counter = 0
        for index, elem_one in enumerate(a):
            logger.info("Event: " + news[index])
            logger.info(elem_one)
            elem_one = reply(elem_one)
            elem_two = b[index]
            logger.info(elem_two)
            elem_two = reply(elem_two)
            if elem_one > elem_two:
                counter += 1
                logger.info(str(elem_one) + "better than " + str(elem_two))
            elif elem_one < elem_two:
                logger.info(str(elem_two) + "less than " + str(elem_one))
                counter -= 1
        return counter

    def read_news(self):
        df = self.news
        currency_one = CROSS[:3]
        currency_second = CROSS[3:6]

        currency_one_data = df[df['Currency'] == currency_one]
        currency_one_data = currency_one_data[(currency_one_data['Volatility'] == 'High Volatility Expected') | (
                    currency_one_data['Volatility'] == 'Moderate Volatility Expected')]
        currency_second_data = df[df['Currency'] == currency_second]
        currency_second_data = currency_second_data[
            (currency_second_data['Volatility'] == 'High Volatility Expected') | (
                        currency_second_data['Volatility'] == 'Moderate Volatility Expected')]
        last_time = parse(self.df['TIME'].iloc[-1])
        currency_second_data = currency_second_data[currency_second_data['Forecast'].notnull()]
        currency_one_data = currency_one_data[currency_one_data['Forecast'].notnull()]
        currency_one_data = currency_one_data[currency_one_data['Time'] <= last_time]
        currency_second_data = currency_second_data[currency_second_data['Time'] <= last_time]
        last_five_1_exp = currency_one_data.tail(25)['Forecast']
        last_five_2_exp = currency_second_data.tail(25)['Forecast']

        last_five_1_act = currency_one_data.tail(25)['Actual']
        last_five_1_news = currency_one_data.tail(25)['Event']
        last_five_2_act = currency_second_data.tail(25)['Actual']
        last_five_2_news = currency_second_data.tail(25)['Event']
        self.news_one = self.get_response(list(last_five_1_act), list(last_five_1_exp), list(last_five_1_news))
        self.news_two = self.get_response(list(last_five_2_act), list(last_five_2_exp), list(last_five_2_news))

        self.buy_fund = False
        self.sell_fund = False
        if self.news_one >= self.news_two:
            logger.info("BUY FUNDAMENTAL")
            self.buy_fund = True
        else:
            logger.info("SELL FUNDAMETAL")
            self.sell_fund = True

    def search_mov_against_short(self):

        self.superata_da_poco_ora_scende_short = False
        self.superata_da_poco_ora_sale_short = False
        self.sopra_ma_avvicinando_short = False
        self.superata_da_poco_ora_sale_short = False
        self.sotto_e_ancora_piu_sotto_short = False
        self.sopra_e_ancora_piu_sopra_short = False
        self.sotto_ma_avvicinando_short = False

        def da_quante_giu(l):
            res = 0
            for i in range(1, 20):
                if l.iloc[-i] < 0:
                    res += 1
                else:
                    break
            return res

        def da_quante_su(l):
            res = 0
            for i in range(1, 20):
                if l.iloc[-i] > 0:
                    res += 1
                else:
                    break
            return res

        def dist_positiva_ma_sta_diminuendo(l):
            dim = False
            for i in range(1, 8):
                if l.iloc[-i] > 0:
                    pass
                else:
                    raise Exception("AO")
            if l.iloc[-1] < l.iloc[-1 - 1] < l.iloc[-1 - 2] < l.iloc[-4] < l.iloc[-5]:
               dim = True
            return dim

        def dist_negativa_e_sta_aumentando(l):
            dim = False
            for i in range(1, 8):
                if l.iloc[-i] < 0:
                    pass
                else:
                    raise Exception("AO neg")
            if l.iloc[-1] < l.iloc[-1 - 1] < l.iloc[-1 - 2]:
                if get_pips(abs(l.iloc[-1])) >= 5.0 and get_pips(abs(l.iloc[-2])) >= 3.0:
                    dim = True
            return dim

        def dist_pos_e_sta_aumentando(l):
            dim = False
            for i in range(1, 8):
                if l.iloc[-i] > 0:
                    pass
                else:
                    raise Exception("AO")
            if l.iloc[-1] > l.iloc[-1 - 1] > l.iloc[-1 - 2]:
                dim = True
            return dim

        def dist_negativa_e_sta_diminuendo(l):
            dim = False
            for i in range(1, 8):
                if l.iloc[-i] < 0:
                    pass
                else:
                    raise Exception("AO neg")
            if l.iloc[-1] > l.iloc[-1 - 1] > l.iloc[-1 - 2]:
                dim = True
            return dim

        HOW_MANY_BEFORE = 15
        moviment = self.df[CROSS + "CLOSE"] - self.df[CROSS + "CLOSE"].rolling(50).mean()

        # Test superata da poco ed era su (ossia ora scende)
        counter = da_quante_giu(moviment)
        if counter <= 3 and counter > 0:
            logger.info("Ha superato da poco, la AVG SHORT, sta scendendo")
            self.superata_da_poco_ora_scende_short = True
            return
        # Test superata da poco ed era giù (ossia ora sale)
        counter = da_quante_su(moviment)
        if counter <= 3 and counter > 0:
            logger.info("Ha superato da poco, la AVG SHORT, sta salendo")
            self.superata_da_poco_ora_sale_short = True
            return

        # Test il prezzo è sopra la AVG ma sta scendendo
        counter = da_quante_su(moviment)
        if counter >= 8 and dist_positiva_ma_sta_diminuendo(moviment):
            logger.info("E' sopra la short AVG, ma si sta avvicinando")
            self.sopra_ma_avvicinando_short = True

        # Test il prezzo è sotto la AVG e si sta allontando ancora
        counter = da_quante_giu(moviment)
        if counter >= 8 and dist_negativa_e_sta_aumentando(moviment):
            logger.info("E' giu la AVG short, e si sta allontandno")
            self.sotto_e_ancora_piu_sotto_short = True

        # Test il prezzo è sopra la AVG e si sta allontandno
        counter = da_quante_su(moviment)
        if counter >= 8 and dist_pos_e_sta_aumentando(moviment):
            logger.info("E' sopra la aVG SHORT e ancora di più")
            self.sopra_e_ancora_piu_sopra_short = True

        # Test il prezzo è sotto la AVG ma si sta avvicinando
        counter = da_quante_giu(moviment)
        if counter >= 8 and dist_negativa_e_sta_diminuendo(moviment):
            logger.info("Il prezzo è sotto la AVG short ma sta salendo")
            self.sotto_ma_avvicinando_short = True


    def search_mov_against_long(self):

        self.superata_da_poco_ora_scende_long = False
        self.superata_da_poco_ora_sale_long = False
        self.sopra_ma_avvicinando_long = False
        self.sopra_e_ancora_piu_sopra_long = False
        self.sotto_ma_avvicinando_long = False
        self.sotto_e_ancora_piu_sotto_long = False

        def da_quante_giu(l):
            res = 0
            for i in range(1, 20):
                if l.iloc[-i] < 0:
                    res += 1
                else:
                    break
            return res

        def da_quante_su(l):
            res = 0
            for i in range(1, 20):
                if l.iloc[-i] > 0:
                    res += 1
                else:
                    break
            return res

        def dist_positiva_ma_sta_diminuendo(l):
            dim = False
            for i in range(1, 8):
                if l.iloc[-i] > 0:
                    pass
                else:
                    raise Exception("AO")
            if l.iloc[-1] < l.iloc[-1 - 1] < l.iloc[-1 - 2] < l.iloc[-3] < l.iloc[-4]:
               dim = True
            return dim

        def dist_negativa_e_sta_aumentando(l):
            dim = False
            for i in range(1, 8):
                if l.iloc[-i] < 0:
                    pass
                else:
                    raise Exception("AO neg")
            if l.iloc[-1] < l.iloc[-1 - 1] < l.iloc[-1 - 2]:
                if get_pips(abs(l.iloc[-1])) >= 5.0 and get_pips(abs(l.iloc[-2])) >= 3.0:
                    dim = True
            return dim

        def dist_pos_e_sta_aumentando(l):
            dim = False
            for i in range(1, 8):
                if l.iloc[-i] > 0:
                    pass
                else:
                    raise Exception("AO")
            if l.iloc[-i] > l.iloc[-1 - 1] > l.iloc[-1 - 2]:
                dim = True
            return dim

        def dist_negativa_e_sta_diminuendo(l):
            dim = False
            for i in range(1, 8):
                if l.iloc[-i] < 0:
                    pass
                else:
                    raise Exception("AO neg")
            if l.iloc[-1] > l.iloc[-1 - 1] > l.iloc[-1 - 2]:
                dim = True
            return dim

        HOW_MANY_BEFORE = 15
        moviment = self.df[CROSS + "CLOSE"] - self.df[CROSS + "CLOSE"].rolling(100).mean()

        # Test superata da poco ed era su (ossia ora scende)
        counter = da_quante_giu(moviment)
        if counter <= 3 and counter > 0:
            logger.info("Ha superato da poco, la AVG Long, sta scendendo")
            self.superata_da_poco_ora_scende_long = True

        # Test superata da poco ed era giù (ossia ora sale)
        counter = da_quante_su(moviment)
        if counter <= 3 and counter > 0:
            logger.info("Ha superato da poco, la AVG Long, sta salendo")
            self.superata_da_poco_ora_sale_long = True


        # Test il prezzo è sopra la AVG ma sta scendendo
        counter = da_quante_su(moviment)
        if counter >= 8 and dist_positiva_ma_sta_diminuendo(moviment):
            logger.info("E' sopra la Long AVG, ma si sta avvicinando")
            self.sopra_ma_avvicinando_long = True

        # Test il prezzo è sotto la AVG e si sta allontando ancora
        counter = da_quante_giu(moviment)
        if counter >= 8 and dist_negativa_e_sta_aumentando(moviment):
            logger.info("E' giu la AVG Long, e si sta allontandno")
            self.sotto_e_ancora_piu_sotto_long = True

        # Test il prezzo è sopra la AVG e si sta allontandno
        counter = da_quante_su(moviment)
        if counter >= 8 and dist_pos_e_sta_aumentando(moviment):
            logger.info("E' sopra la aVG Long e ancora di più")
            self.sopra_e_ancora_piu_sopra_long = True

        # Test il prezzo è sotto la AVG ma si sta avvicinando
        counter = da_quante_giu(moviment)
        if counter >= 8 and dist_negativa_e_sta_diminuendo(moviment):
            logger.info("Il prezzo è sotto la AVG Long ma sta salendo")
            self.sotto_ma_avvicinando_long = True

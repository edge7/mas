from sklearn.linear_model import LogisticRegression

from constants.constants import PATH
from market_info.market_info import CROSS, in_pips, get_pips
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from utility.adjust import adjust_data
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from utility.discovery import TP, SL, chart, che_ha_fatto, scalp_man, YOCLOSE
from sklearn.model_selection import GridSearchCV


class GridSearchCustomModel(object):
    model = None
    params = None

    def __init__(self, model, params):
        self.model = model
        self.params = params


def do_grid_search(list_models, X, y):
    to_return = []
    print("Shape X \n")
    print(X.shape)

    for model in list_models:
        print("Grid Search running for model \n")
        print(model.model)
        print("  ...  \n")
        clf = GridSearchCV(model.model, model.params, cv=5, n_jobs=-1,
                           verbose=1, scoring='f1_weighted')
        clf.fit(X, y)
        print(" printing best parameters: \n")
        print(clf.best_params_)
        print(" Print best score \n")
        print(clf.best_score_)
        to_return.append(clf)

    return to_return


count = {}
d = {}
def update_fi(fi):

    l = []
    global count
    for feature, value in fi:
        old_v = d.get(feature, 0.0)
        old_v += value
        d[feature] = old_v
        c = count.get(feature, 1)
        l.append((feature, old_v/c))
        c +=1
        count[feature] = c
    return l


def compute_avg_time(x):
    l = list(x)
    mi = min(l)
    ma = max(l)
    ranges = ((ma - mi) / 15.0)
    ra = []
    start = mi
    for i in range(15):
        ra.append((start, start + ranges))
        start = start + ranges

    def find_range_index(item,ra):
        for index, i in enumerate(ra):
            s = i[0]
            e = i[1]
            if item >= s and item < e:
                return index

        return index

    return find_range_index(l[-1], ra)


def harmonic(x):
    x = list(x)
    len_ = len(x) *1.0
    su = 0
    for i in x:
        su += (1/i)
    return len_ / su


def create_random_forest(df, news, result, normal):
    # Create target
    df_copy = df.copy()
    df_copy = df.merge(news, how='left', left_on="TIME", right_on="Time")
    del df_copy['Time']
    df_copy = df_copy.fillna(method='ffill')
    df_copy = df_copy.dropna(axis=1, how='all')

    for i in df_copy.index:
        i = i + 1
        if i <= len(result):
            continue
        tmp = df.iloc[0:i]
        if tmp.empty:
            continue
        try:
            trigger, tp, sl, action = scalp_man(tmp)
            res = action
        except:
            res = None
        if res is None:
            res = "REMOVE"
        else:
            pass
        if res == "SELL":
            res = -1
        if res == "BUY":
            res = 1
        result.append(res)
    df_copy['filter'] = pd.Series(result)
    df_copy['TARGET'] = - df_copy[CROSS + "CLOSE"] + df_copy[
        CROSS + "CLOSE"].shift(-YOCLOSE)
    df_copy['BODY_IN_PIPS'] = df_copy[CROSS + "CLOSE"] - df_copy[
        CROSS + "OPEN"]

    #for i in range(1, 3):
    #    df_copy['BODY_IN_PIPS' + str(i)] = df_copy['BODY_IN_PIPS'].shift(i)
    #    df_copy['BODY_IN_PIPS_DIFF' + str(i)] = df_copy['BODY_IN_PIPS'].diff(periods= i)
    high = []
    low = []
    # for index, row in df_copy.iterrows():
    #     body = row['BODY_IN_PIPS']
    #     open = row[CROSS + "OPEN"]
    #     close = row[CROSS + "CLOSE"]
    #     low_ = row[CROSS + "LOW"]
    #     high_ = row[CROSS + "HIGH"]
    #     if body >= 0:
    #         high.append(high_ - close)
    #         low.append(open - low_)
    #     else:
    #         high.append(high_ - open)
    #         low.append(close - low_)

    # low_pips = pd.Series(low)
    # high_pips = pd.Series(high)

    # df_copy['LOW_P'] = low_pips
    # df_copy['HIGH_P'] = high_pips

    df_copy['AVG_200'] = df_copy[CROSS + "CLOSE"] - df_copy[
        CROSS + "CLOSE"].ewm(span=200).mean()
    df_copy['AVG_50'] = df_copy[CROSS + "CLOSE"] - df_copy[
        CROSS + "CLOSE"].ewm(span=50).mean()
    df_copy['AVG_100'] = df_copy[CROSS + "CLOSE"] - df_copy[
        CROSS + "CLOSE"].ewm(span=100).mean()
    df_copy['AVG_300'] = df_copy[CROSS + "CLOSE"] - df_copy[
        CROSS + "CLOSE"].ewm(span=300).mean()
    df_copy['RANGE_150'] = df_copy[CROSS + "CLOSE"].rolling(window=150).max() - \
                           df_copy[CROSS + "CLOSE"].rolling(window=150).min()
    df_copy['RANGE_200'] = df_copy[CROSS + "CLOSE"].rolling(window=200).max() - \
                           df_copy[CROSS + "CLOSE"].rolling(window=200).min()
    df_copy['RANGE_100'] = df_copy[CROSS + "CLOSE"].rolling(window=100).max() - \
                           df_copy[CROSS + "CLOSE"].rolling(window=100).min()
    df_copy['RANGE_50'] = df_copy[CROSS + "CLOSE"].rolling(window=50).max() - \
                          df_copy[CROSS + "CLOSE"].rolling(window=50).min()
    df_copy['AVG_50_SUM'] = (
            df_copy[CROSS + "CLOSE"] - df_copy[CROSS + "CLOSE"].ewm(
        span=50).mean()).rolling(window=100).sum()
    df_copy['AVG_100_SUM'] = (
            df_copy[CROSS + "CLOSE"] - df_copy[CROSS + "CLOSE"].ewm(
        span=100).mean()).rolling(window=100).sum()
    df_copy['AVG_300_SUM'] = df_copy['AVG_300'].rolling(window=100).sum()
    df_copy['AVG_200_SUM'] = df_copy['AVG_200'].rolling(window=100).sum()

    df_copy['AVG_50_SUM_DIFF'] = df_copy['AVG_50_SUM'].diff(periods=40)
    df_copy['AVG_100_SUM_DIFF'] = df_copy['AVG_100_SUM'].diff(periods=40)
    for col in list(df_copy.columns):
        if 'FUND_' in col and False:
            df_copy[col + "AVG_50"] = df_copy[col].rolling(window = 50).mean()
            df_copy[col + "AVG_100"] = df_copy[col].rolling(window=100).mean()
    #df_copy['STD_5'] = df_copy['BODY_IN_PIPS'].rolling(window=5).std()
    #df_copy['STD_10'] = df_copy['BODY_IN_PIPS'].rolling(window=10).std()
    #df_copy['AVG_ABS_5'] = df_copy['BODY_IN_PIPS'].apply(lambda x: abs(x)).rolling(
    #    window=5).mean()
    #df_copy['AVG_5'] = df_copy['BODY_IN_PIPS'].rolling(window=5).mean()
    #df_copy[CROSS + "HIGHP1"] = df_copy[CROSS + "HIGH"] - df_copy[CROSS + "CLOSE"]
    #df_copy[CROSS + "LOWP1"] = - df_copy[CROSS + "LOW"] + df_copy[CROSS + "CLOSE"]
    #df_copy["DIST_MIN_15"] = df_copy[CROSS + "CLOSE"] - df_copy[CROSS+"CLOSE"].rolling(window=25).min()
    #df_copy["DIST_MAX_15"] = df_copy[CROSS + "CLOSE"] - df_copy[CROSS+"CLOSE"].rolling(window=25).max()

    def avg_gain(l):
        g = 0.00001
        for i in l:
            if i > 0:
                g+= i
        return g

    def avg_loss(l):
        g = 0.00001
        for i in l:
            if i < 0:
                g += abs(i)
        return g

    #df_copy["AVG_GAIN_14"] = df_copy["BODY_IN_PIPS"].rolling(window=14).apply(lambda l: avg_gain(l)) / 14
    #df_copy["AVG_LOSS_14"] = df_copy["BODY_IN_PIPS"].rolling(window=14).apply( lambda l: avg_loss(l)) / 14
    #df_copy["RSI_14"] = 100 - 100 /(1 + df_copy["AVG_GAIN_14"] / df_copy["AVG_LOSS_14"])
    #df_copy["HISTOGRAM_100"]= df_copy[CROSS + "CLOSE"].rolling(window = 100).apply(lambda x: compute_avg_time(x))
    #df_copy["HISTOGRAM_200"]= df_copy[CROSS + "CLOSE"].rolling(window = 200).apply(lambda x: compute_avg_time(x))
    #df_copy["HISTOGRAM_300"]= df_copy[CROSS + "CLOSE"].rolling(window = 300).apply(lambda x: compute_avg_time(x))
    #df_copy["DIST_HARMNIC_50"] = df_copy[CROSS + "CLOSE"] - df_copy[CROSS + "CLOSE"].rolling(window = 50).apply(lambda x: harmonic(x))
    #df_copy["DIST_HARMNIC_100"] = df_copy[CROSS + "CLOSE"] - df_copy[CROSS + "CLOSE"].rolling(window = 100).apply(lambda x: harmonic(x))
    #df_copy["DIST_HARMNIC_200"] = df_copy[CROSS + "CLOSE"] - df_copy[CROSS + "CLOSE"].rolling(window = 200).apply(lambda x: harmonic(x))

    for window in [25, 50, 100, 200, 300]:
        for lag in [5, 10, 20, 50, 80]:
            if window <= lag:
                continue
            df_copy['AUTOCORR_' + str(window) + "_" + str(lag)] = df_copy['BODY_IN_PIPS'].rolling(window).apply(lambda x: x.autocorr(lag), raw=False)
            df_copy['AUTOCORRClose_' + str(window) + "_" + str(lag)] = df_copy[CROSS + "CLOSE"].rolling(window).apply(
                lambda x: x.autocorr(lag), raw=False)




    # df_copy['SUMAVG2'] = df_copy['AVG'].rolling(window=30).sum()
    # for i in range(1,2):
    #    df_copy['AVG' + str(i)] = df_copy['AVG'].shift(i)
    #    df_copy['SUMAVG' + str(i)] = df_copy['SUMAVG'].shift(i)
    # df_copy['AVGS'] = df_copy['AVG'].shift(1)

    # df_copy['AVG_D'] = df_copy['AVG'] - df_copy['AVGS']
    # df_copy = df_copy[df_copy['AVG'] * df_copy['AVG'].shift(1) < 0]
    # del df_copy['AVGS']
    df_copy['H'] = df_copy['TIME'].apply(lambda x: x.hour)
    del df_copy[CROSS + 'OPEN']
    del df_copy[CROSS + 'CLOSE']
    del df_copy[CROSS + 'HIGH']
    del df_copy[CROSS + 'LOW']
    # del df_copy['']
    del df_copy['H']
    del df_copy['TIME']

    def ap_target(x):
        if np.isnan(x): return x
        x = get_pips(x)
        if x >= 0:
            return "BUY"
        if x <= -0:
            return "SELL"
        return "OUT"

    df_copy['TARGET'] = df_copy['TARGET'].apply(lambda x: ap_target(x))
    assert df_copy.tail(1)['filter'].iloc[-1] != "REMOVE"
    df_copy = df_copy[df_copy['filter'] != "REMOVE"]
    del df_copy['filter']
    df_copy.reset_index(drop=True)
    # df_copy['BODY_IN_PIPS']
    to_predict = df_copy.tail(1)
    df_copy = df_copy.dropna()
    df_copy.reset_index(drop=True)
    percent = df_copy.shape[0] * 0.1
    percent = int(percent)
    train = df_copy.head(df_copy.shape[0] - percent)
    test = df_copy.tail(percent)

    target_train = train['TARGET']
    target_test = test['TARGET']

    del train['TARGET']
    del test['TARGET']
    param_grid_log_reg = {'C': 2.0 ** np.arange(-7, 6), 'penalty':['l1','l2']}
    gdLog = GridSearchCustomModel(LogisticRegression(penalty='l1', max_iter=200, random_state=42, n_jobs=-1),
                                  param_grid_log_reg)

    param_grid_rf = {'n_estimators': [100, 200, 300], 'max_depth': [4, 7, 12]}
    gdRf = GridSearchCustomModel(
        RandomForestClassifier(n_jobs=-1, random_state=42), param_grid_rf)

    clf = do_grid_search([gdRf], train,
                         target_train.values.ravel())[0].best_estimator_
    lr = do_grid_search([gdLog], train,
                        target_train.values.ravel())[0].best_estimator_

    lr.fit(train, target_train)
    clf.fit(train, target_train)
    t = clf.predict(test)
    f = f1_score(target_test, t, average='weighted')
    fi = list(zip(train.columns, clf.feature_importances_))
    fi = update_fi(fi)
    fi.sort(key=lambda x: x[1], reverse=True)
    c = confusion_matrix(target_test, t)
    print(c)
    print(fi)
    print(f)
    print("\n\n      LR\n\n")
    t = lr.predict(test)
    fl = f1_score(target_test, t, average='weighted')
    c = confusion_matrix(target_test, t)
    print(c)
    print(fl)
    del to_predict['TARGET']
    if f > fl:
        res = clf.predict(to_predict)[-1]
        print("USING RF: "+ res)
    else:
        res = lr.predict(to_predict)[-1]
        print("USING LR: " + res)
    ret = zip(clf.classes_, clf.predict_proba(to_predict)[0])
    ret = list(ret)
    print(ret)
    # d = {}
    # for elem in ret:
    #     d[elem[0]] = elem[1]
    #
    # if normal == "SELL":
    #     if d[normal] > d["BUY"]:
    #         res = "SELL"
    #
    # if normal == "BUY":
    #     if d[normal] > d["SELL"]:
    #         res = "BUY"

    return res, result
df = pd.read_csv(PATH + 'o.csv', sep=",").reset_index(drop=True)
df = adjust_data(df, CROSS, candle=4)


# chart(df)

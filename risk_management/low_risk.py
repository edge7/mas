def get_last_different(bals, quanti=5):
    l = len(bals)
    diffs = []
    just_values = []
    for i in range(1, l):
        t = bals[-i][0]
        v = bals[-i][1]
        if v not in just_values:
            just_values.append(v)
            diffs.append((t, float(v)))
            if len(diffs) == quanti:
                break
    return diffs


def apply_diffs(diffs):
    to_return = []
    for index, item in enumerate(diffs):
        try:
            to_return.append(diffs[index] - diffs[index + 1])
        except:
            pass
    return to_return


def count_pos_neg(values):
    ret = 0
    for v in values:
        if v > 0:
            ret += 1
        if v < 0:
            ret -= 1
    return ret


def risk_management(balances, posiz=2.0, max_pos=10.0, min_pos=0.1):
    QUANTI = 10
    balances = [(b.split(',')[0], b.split(',')[1]) for b in balances]
    diffs = get_last_different(balances, quanti=10)
    if len(diffs) < QUANTI:
        return 1.0
    diff_values = [d[1] for d in diffs]
    vs = apply_diffs(diff_values)

    gradient = float(count_pos_neg(vs))
    if gradient < 0:
        pos = (len(vs) + (gradient)) / len(vs)
        pos = pos * posiz

    if gradient > 0:
        pos = (len(vs) + 1.1 * abs(gradient)) / len(vs)
        pos = pos * posiz
    if gradient == 0:
        pos = 1.0

    pos = round(pos, 2)
    if pos > max_pos:
        pos = max_pos
    if pos < min_pos:
        pos = min_pos
    print("Position has been scaled to {} as gradient is ".format(str(pos)))
    print("Gradient is " + str(gradient))
    return round(pos, 2)



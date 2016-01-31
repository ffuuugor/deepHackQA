__author__ = 'ffuuugor'
import pandas
import os
from collections import Counter
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np

def vote_best(df, *dfs):
    """
    Each approach votes for a single answer. In case of a tie, more left df is considered as more reliable
    :param dfs:
    :return: df [id, ans]
    """
    res_df = predict_one(df)
    for other_df in dfs:
        res_df = res_df.merge(predict_one(other_df), left_index=True, right_index=True)

    def pick_one(serie):
        counts = serie.value_counts()
        maxval = counts.max()

        maxvals = counts[counts == maxval]
        for i in range(0, 4):
            if serie[i] in maxvals:
                return serie[i]

    return res_df.apply(pick_one, axis=1)

def vote_ranks(df, *dfs):
    """
    Each approach votes for all 4 answers: 3 points for 1st place, 2 for 2nd, etc
    :param dfs:
    :return: df [id, ans]
    """

    def points(serie):
        #I would've users argsort() instead, but it behaves really strange
        slist = sorted(serie.to_dict().items(), key=lambda x: x[1])
        slist = map(lambda x: x[0], slist)

        ddict = dict(zip(slist, range(0, len(slist))))
        return pandas.Series(ddict)

    def make_rank(df):
        return df.apply(points, axis=1)

    res_df = make_rank(df)
    for other_df in dfs:
        res_df = res_df.merge(make_rank(other_df), left_index=True, right_index=True)
    #
    def sum_by_label(serie):
        pairs = serie.to_dict().items()
        pairs = map(lambda x: (x[0][:1], x[1]), pairs)

        sumscores = Counter()
        for label, score in pairs:
            sumscores[label] += score

        return sumscores.most_common()[0][0]

    return res_df.apply(sum_by_label, axis=1)

def vote_probability(df, *dfs):
    """
    Each approach votes with normalized score
    :param dfs:
    :return: df [id, ans]
    """

    def points(serie):
        score_sum = serie.sum()
        return serie/(score_sum+0.01)

    def make_rank(df):
        return df.apply(points, axis=1)

    res_df = make_rank(df)
    for other_df in dfs:
        res_df = res_df.merge(make_rank(other_df), left_index=True, right_index=True)

    def sum_by_label(serie):
        pairs = serie.to_dict().items()
        pairs = map(lambda x: (x[0][:1], x[1]), pairs)

        sumscores = Counter()
        for label, score in pairs:
            sumscores[label] += score

        return sumscores.most_common()[0][0]

    return res_df.apply(sum_by_label, axis=1)

def avg_correct_probability(df_real, df_scores):
    def points(serie):
        serie = serie - serie.min()
        score_sum = serie.sum()
        return serie/score_sum

    df_scores = df_scores.apply(points, axis=1).join(df_real)
    def corrProb(serie):
        correct = serie["correctAnswer"]
        return serie[correct]

    return df_scores.apply(corrProb, axis=1).mean()

def vote_success_zone(*pairs):
    """
    Vote with respect to given polynomials
    :param pairs: (df, poly)
    :return:
    """
    def points(serie):
        serie = serie.apply(lambda x: 0 if x < 0 else x)
        score_sum = serie.sum()
        if score_sum == 0:
            return 0
        else:
            return serie/score_sum

    dfs = []
    for df, poly in pairs:
        def pick_confidence(serie):
            ans = serie.argmax()
            probability = serie.max()
            confidence = poly(probability)

            return ans, confidence

        dfs.append(df.apply(points, axis=1).apply(pick_confidence, axis=1))

    res_df = dfs[0].to_frame()
    for i in range(1, len(dfs)):
        res_df = res_df.merge(dfs[i].to_frame(), left_index=True, right_index=True)


    res_df.columns=[str(i) for i in range(0,len(pairs))]
    def pick_one(serie):
        values = serie.to_dict().items()
        values = map(lambda x: (x[0],x[1][0],x[1][1]), values)
        return sorted(values, key=lambda x: x[2], reverse=True)[0]

    return res_df.apply(pick_one, axis=1)

def success_zone_cumulative(df_real, df_scores):
    """
    Approximates method confidence based on cumulative precision
    :param df_real:
    :param df_scores:
    :return: polynomial
    """
    def points(serie):
        serie = serie.apply(lambda x: 0 if x < 0 else x)
        score_sum = serie.sum()
        if score_sum == 0:
            return 0
        else:
            return serie/score_sum

    def pick(x):
        return pandas.Series({"answer":x.argmax(), "prob":x.max()})

    df_scores = df_scores.apply(points, axis=1).apply(pick, axis=1)

    records = df_scores.join(df_real).to_records()
    records = sorted(records, key=lambda x: x[2], reverse=True)


    precision_tuples = []

    total_cnt = 0
    correct_cnt = 0
    for id, ans, prob, correct in records:
        if ans == correct:
            correct_cnt += 1

        total_cnt += 1
        precision_tuples.append((correct_cnt, total_cnt, prob))


    def remove_duplicates(llist):
        res = [llist[len(llist)-1]]
        for i in range(len(llist)-2, -1, -1):
            if llist[i+1][2] != llist[i][2]:
                res.append(llist[i])

        res.reverse()
        return res

    precision_tuples = remove_duplicates(precision_tuples)

    to_plot_y = map(lambda x: float(x[0])/x[1], precision_tuples)
    to_plot_x = map(lambda x: x[2], precision_tuples)

    p3 = np.poly1d(np.polyfit(to_plot_x, to_plot_y, 10))
    return p3

def success_zone_group(df_real, df_scores):
    """
    Approximates method confidence based on average precision
    :param df_real:
    :param df_scores:
    :return: poly
    """
    def points(serie):
        serie = serie.apply(lambda x: 0 if x < 0 else x)
        score_sum = serie.sum()
        if score_sum == 0:
            return 0
        else:
            return serie/score_sum

    def pick(x):
        return pandas.Series({"answer":x.argmax(), "prob":x.max()})

    df_scores = df_scores.apply(points, axis=1).apply(pick, axis=1)

    records = df_scores.join(df_real).to_records()
    records = sorted(records, key=lambda x: x[2], reverse=True)[300:-300]

    precision_tuples = []
    for id, ans, prob, correct in records:
        if ans == correct:
            precision_tuples.append((True, prob))
        else:
            precision_tuples.append((False, prob))

    to_plot_x = []
    to_plot_y = []
    for i in range(0,len(precision_tuples), 20):
        sublist = precision_tuples[i:i+20]
        avg_probability = np.mean(map(lambda x: x[1], sublist))
        avg_precision = float(len(filter(lambda x: x[0]==True, sublist)))/len(sublist)

        to_plot_x.append(avg_probability)
        to_plot_y.append(avg_precision)

    p3 = np.poly1d(np.polyfit(to_plot_x, to_plot_y, 3))
    return p3

    # traces = []
    # traces.append(go.Scatter(
    #         x = to_plot_x,
    #         y = to_plot_y
    #     ))
    #
    # traces.append(go.Scatter(
    #         x = to_plot_x,
    #         y = [p3(x) for x in to_plot_x],
    #     ))
    #
    # plotly.offline.plot(traces)

def plot_polys(polys):
    traces = []
    x_axis = np.linspace(0,1,1000)

    for poly in polys:
        y_axis = [poly(x) for x in x_axis]

        traces.append(go.Scatter(
            x = x_axis,
            y = y_axis
        ))
    plotly.offline.plot(traces)

def precision(df_real, df_predicted):
    """
    Share of correct answers
    :param df_real:
    :param df_predicted:
    :return: float
    """

    merged = df_real.merge(df_predicted, left_index=True, right_index=True)
    flags = merged.iloc[:,0] == merged.iloc[:,1]

    # return flags
    return float(len(flags[flags == True]))/len(flags)

def predict_one(df):
    """
    Pick most probable answer
    :param df: df [id,A,B,C,D]
    :return: df [id, ans]
    """
    return pandas.DataFrame(df.apply(lambda x: x.argmax(), axis=1), columns=["answer"])


if __name__ == '__main__':
    df_tema = pandas.DataFrame.from_csv(os.path.join("data","predictions","tema_scores3.csv"), index_col=4).fillna(0)
    df_es = pandas.DataFrame.from_csv(os.path.join("data","predictions","es_scores.csv")).fillna(0)
    df_snowball = pandas.DataFrame.from_csv(os.path.join("data","predictions","snowball_scores.csv")).fillna(0)
    df_ck = pandas.DataFrame.from_csv(os.path.join("data","predictions","ck12_scores.csv"), index_col=4).fillna(0)
    df_glove = pandas.DataFrame.from_csv(os.path.join("data","predictions","glove_scores.csv"), index_col=4).fillna(0)
    df_real = pandas.DataFrame.from_csv(os.path.join("data","training_set_answers.csv")).fillna(0)

    df_es_valid = pandas.DataFrame.from_csv(os.path.join("data","predictions","es_scores_validation.csv")).fillna(0)
    df_snowball_valid = pandas.DataFrame.from_csv(os.path.join("data","predictions","snowball_scores_validation.csv")).fillna(0)
    df_ck_valid = pandas.DataFrame.from_csv(os.path.join("data","predictions","ck12_scores_validation.csv"), index_col=4).fillna(0)
    df_glove_valid = pandas.DataFrame.from_csv(os.path.join("data","predictions","glove_scores_validation.csv"), index_col=4).fillna(0)

    df_vote_one = vote_best(df_tema, df_snowball, df_es, df_ck, df_glove).to_frame()
    df_vote_four = vote_ranks(df_tema, df_snowball, df_es, df_ck, df_glove).to_frame()
    df_vote_prob = vote_probability(df_tema, df_snowball, df_es, df_ck, df_glove).to_frame()

    print df_tema
    dfs = [df_tema, df_snowball, df_es, df_ck, df_glove]
    polys = [success_zone_group(df_real, df) for df in dfs]
    df_vote_confidence = vote_success_zone(*zip(dfs, polys))
    print df_vote_confidence.apply(lambda x: x[2]).mean()
    print df_vote_confidence.apply(lambda x: x[2]).median()
    #
    print "tema", precision(df_real, predict_one(df_tema))
    print "snowball", precision(df_real, predict_one(df_snowball))
    print "es", precision(df_real, predict_one(df_es))
    print "ck", precision(df_real, predict_one(df_ck))
    print "glove", precision(df_real, predict_one(df_glove))

    print "vote one", precision(df_real, df_vote_one)
    print "vote four", precision(df_real, df_vote_four)
    print "vote prob", precision(df_real, df_vote_prob)
    print "vote_success", precision(df_real, df_vote_confidence.apply(lambda x: x[1]).to_frame())

    # valid_dfs = [df_snowball_valid, df_es_valid, df_ck_valid, df_glove_valid]
    # df_vote_one_valid = vote_success_zone(*zip(valid_dfs, polys)).to_frame()
    # df_vote_one_valid.to_csv("submission_votefour.csv")

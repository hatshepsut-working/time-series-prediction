'''
Created on May 15, 2018

@author: zjn
'''
from math import sqrt
import csv
from operator import itemgetter, attrgetter
from numpy import *
from sklearn.preprocessing import normalize

from general_functions import *
from input_data import *


def normalize_matrix_by_row(input_matrix):
    return 1;


def main():
    before_time_series = []
    input_time_series = []
    five = []
    csv_file = csv.reader(open('moisture.csv','r'))
    print(csv_file)
    for mois in csv_file:
        for value in mois:
            before_time_series.append(int(value))
    print(before_time_series)
    for i in range(len(before_time_series)):
        if len(input_time_series) >= 350:
            break
        if i % 5 == 0 and i != 0:
            five.sort()
            input_time_series.append(five[2])
            five.clear()
            five.append(before_time_series[i])
        else:
            five.append(before_time_series[i])
    partition_size = 10;
    umin = math.floor(min(input_time_series) / 10) * 10;
    umax = math.ceil(max(input_time_series) / 10) * 10;
    #umin = math.floor(min(input_time_series) / 1000.0) * 1000;
    #umax = math.ceil(max(input_time_series) / 1000.0) * 1000;
    universe = (umin, umax);
    print(universe);
    # 2: Partition of universe
    # Method: Dividing in the half-thousands
    nIter = int((umax - umin) / partition_size);
    print(nIter);
    u_vectorized = [];
    for i in range(nIter):
        u_vectorized.append((umin + i * partition_size, umin + (i + 1) * partition_size));
    print(u_vectorized);
    # 3: Analyse historical data, putting its values in the intervals
    historical_data_fuzzified = [];
    for val in input_time_series:
        historical_data_fuzzified.append(fetch_fuzzy_class(val, u_vectorized));
    print(historical_data_fuzzified);
    # 4: Establish the relations between fuzzy classes
    # In Lee predictor, number of occurrences and their chronological order are relevant,
    # which is why duplicates are not simply removed
    historical_relations_fuzzy = [];
    historical_weights = {};
    for i in range(len(historical_data_fuzzified) - 1):
        _pair = (historical_data_fuzzified[i].get('fuzzy_class'), historical_data_fuzzified[i + 1].get('fuzzy_class'));
        historical_weights[_pair] = i;
        historical_relations_fuzzy.append(
            (historical_data_fuzzified[i].get('fuzzy_class'), historical_data_fuzzified[i + 1].get('fuzzy_class')));

    historical_relations_fuzzy = sorted(historical_relations_fuzzy, key=itemgetter(0, 1));
    historical_relations_fuzzy_weighted = [(x, float(historical_relations_fuzzy.count(x))) for x in
                                           historical_relations_fuzzy];
    historical_relations_fuzzy_weighted = sorted(list(set(historical_relations_fuzzy_weighted)), key=itemgetter(0, 1));

    # This could be activated, although for Alabama Enrollment time series it decreased
    # the predictor accuracy
    # print historical_relations_fuzzy_weighted;
    # historical_relations_fuzzy_weighted = [(x[0],x[1]+historical_weights[x[0]]) for x in historical_relations_fuzzy_weighted];
    # print historical_relations_fuzzy_weighted;
    # print _teste;

    # Implementation of Qiu et al. generalization of Lee et al. predictor, as described in
    # 'A generalized method for forecasting based on fuzzy time series'
    # by Wangren Qiu, Xiaodong Liu, Hailin Li

    # Algorithm parameters
    alfa_param = 1;
    l_param = 2;

    # Build and normalize pertinence matrix
    pertinence_matrix = [];
    for j in range(len(input_time_series)):
        ts_val = input_time_series[j];
        _currently_fuzzy = [];
        for i in range(len(u_vectorized)):
            # needs to include alfa parameter in consideration
            _currently_fuzzy.append((fuzzy_pertinence_in_tuple(ts_val, u_vectorized[i], u_vectorized)) ** alfa_param);
        pertinence_matrix.append(_currently_fuzzy);

    # consider only l_param fuzzy sets on prediction
    pertinence_matrix = generalize_discard_irrelevant_values(pertinence_matrix, l_param);
    # print pertinence_matrix[5:6];
    pertinence_matrix = mat(pertinence_matrix);
    # print pertinence_matrix;
    normalized_pertinence_matrix = normalize(pertinence_matrix, norm='l1', axis=1);
    # print normalized_pertinence_matrix;

    # process relation matrix
    nPartitions = len(u_vectorized);
    R_lee = zeros([nPartitions, nPartitions]);
    for i in historical_relations_fuzzy_weighted:
        _tuple = i[0];
        R_lee[_tuple[0]][_tuple[1]] = i[1];
    for i in range(len(R_lee)):
        val = R_lee[i];
        if any(val) == False:
            R_lee[i][i] = 1;
    R_lee_normalized = normalize(R_lee, norm='l1', axis=1);
    print(R_lee_normalized);

    # Generating midpoint vector
    midpoint_vector = [];
    for i in u_vectorized:
        midpoint_vector.append(get_midpoint(i));
    # print midpoint_vector;

    """
      Prediction algorithm
    """
    for i in range(len(normalized_pertinence_matrix) - 1):
        # print normalized_pertinence_matrix[i:i+1];
        # print (R_lee_normalized*(mat(midpoint_vector)).T);
        # forecasted_data.append(float(normalized_pertinence_matrix[i:i+1]*(R_lee_normalized*(mat(midpoint_vector)).T)));
        historical_data_fuzzified[i + 1]['forecasted_data'] = float(
            normalized_pertinence_matrix[i:i + 1] * (R_lee_normalized * (mat(midpoint_vector)).T));

    print(historical_data_fuzzified)
    print(len(historical_data_fuzzified))

    # evaluation: MSE RMSE MAE
    error = []
    squaredError = []
    absError = []
    i = len(historical_data_fuzzified)
    for z in range(1, i):
        error.append(
            historical_data_fuzzified[z]['forecasted_data'] - historical_data_fuzzified[z]['actual_data'])
    for val in error:
        squaredError.append(val * val)
        absError.append(abs(val))

    # MSE：均方误差
    print("MSE = ", sum(squaredError) / len(squaredError))

    # RMSE：均方根误差
    print("RMSE = ", sqrt(sum(squaredError) / len(squaredError)))

    # MAE：平均绝对误差
    print("MAE = ", sum(absError) / len(absError))

        # Graph Plotting
    plot_comparison_graph(historical_data_fuzzified, 'Qiu\'s model', ['actual', 'forecasted']);

'''def main():
    # 1: Define the universe of discourse
    # Method: Round min and max to thousand
    partition_size = 1000;
    umin = math.floor(min(input_time_series) / 1000.0) * 1000;
    umax = math.ceil(max(input_time_series) / 1000.0) * 1000;
    universe = (umin, umax);
    print(universe);
    # 2: Partition of universe
    # Method: Dividing in the half-thousands
    nIter = int((umax - umin) / partition_size);
    print(nIter);
    u_vectorized = [];
    for i in range(nIter):
        u_vectorized.append((umin + i * partition_size, umin + (i + 1) * partition_size));
    print(u_vectorized);
    # 3: Analyse historical data, putting its values in the intervals
    historical_data_fuzzified = [];
    for val in input_time_series:
        historical_data_fuzzified.append(fetch_fuzzy_class(val, u_vectorized));
    print(historical_data_fuzzified);
    # 4: Establish the relations between fuzzy classes
    # In Lee predictor, number of occurrences and their chronological order are relevant,
    # which is why duplicates are not simply removed
    historical_relations_fuzzy = [];
    historical_weights = {};
    for i in range(len(historical_data_fuzzified) - 1):
        _pair = (historical_data_fuzzified[i].get('fuzzy_class'), historical_data_fuzzified[i + 1].get('fuzzy_class'));
        historical_weights[_pair] = i;
        historical_relations_fuzzy.append(
            (historical_data_fuzzified[i].get('fuzzy_class'), historical_data_fuzzified[i + 1].get('fuzzy_class')));

    historical_relations_fuzzy = sorted(historical_relations_fuzzy, key=itemgetter(0, 1));
    historical_relations_fuzzy_weighted = [(x, float(historical_relations_fuzzy.count(x))) for x in
                                           historical_relations_fuzzy];
    historical_relations_fuzzy_weighted = sorted(list(set(historical_relations_fuzzy_weighted)), key=itemgetter(0, 1));

    # This could be activated, although for Alabama Enrollment time series it decreased
    # the predictor accuracy
    # print historical_relations_fuzzy_weighted;
    # historical_relations_fuzzy_weighted = [(x[0],x[1]+historical_weights[x[0]]) for x in historical_relations_fuzzy_weighted];
    # print historical_relations_fuzzy_weighted;
    # print _teste;

    # Implementation of Qiu et al. generalization of Lee et al. predictor, as described in
    # 'A generalized method for forecasting based on fuzzy time series'
    # by Wangren Qiu, Xiaodong Liu, Hailin Li

    # Algorithm parameters
    alfa_param = 1;
    l_param = 2;

    # Build and normalize pertinence matrix
    pertinence_matrix = [];
    for j in range(len(input_time_series)):
        ts_val = input_time_series[j];
        _currently_fuzzy = [];
        for i in range(len(u_vectorized)):
            # needs to include alfa parameter in consideration
            _currently_fuzzy.append((fuzzy_pertinence_in_tuple(ts_val, u_vectorized[i], u_vectorized)) ** alfa_param);
        pertinence_matrix.append(_currently_fuzzy);

    # consider only l_param fuzzy sets on prediction
    pertinence_matrix = generalize_discard_irrelevant_values(pertinence_matrix, l_param);
    # print pertinence_matrix[5:6];
    pertinence_matrix = mat(pertinence_matrix);
    # print pertinence_matrix;
    normalized_pertinence_matrix = normalize(pertinence_matrix, norm='l1', axis=1);
    # print normalized_pertinence_matrix;

    # process relation matrix
    nPartitions = len(u_vectorized);
    R_lee = zeros([nPartitions, nPartitions]);
    for i in historical_relations_fuzzy_weighted:
        _tuple = i[0];
        R_lee[_tuple[0]][_tuple[1]] = i[1];
    for i in range(len(R_lee)):
        val = R_lee[i];
        if any(val) == False:
            R_lee[i][i] = 1;
    R_lee_normalized = normalize(R_lee, norm='l1', axis=1);
    print(R_lee_normalized);

    # Generating midpoint vector
    midpoint_vector = [];
    for i in u_vectorized:
        midpoint_vector.append(get_midpoint(i));
    # print midpoint_vector;

    """
      Prediction algorithm
    """
    for i in range(len(normalized_pertinence_matrix) - 1):
        # print normalized_pertinence_matrix[i:i+1];
        # print (R_lee_normalized*(mat(midpoint_vector)).T);
        # forecasted_data.append(float(normalized_pertinence_matrix[i:i+1]*(R_lee_normalized*(mat(midpoint_vector)).T)));
        historical_data_fuzzified[i + 1]['forecasted_data'] = float(
            normalized_pertinence_matrix[i:i + 1] * (R_lee_normalized * (mat(midpoint_vector)).T));

        # Graph Plotting
    plot_comparison_graph(historical_data_fuzzified, 'Qiu, Liu e Li', ['actual', 'forecasted']);

'''
if __name__ == '__main__':
    main();
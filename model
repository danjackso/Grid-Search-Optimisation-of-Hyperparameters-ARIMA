from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from pandas import read_csv
import itertools
import numpy as np
import operator

# Making a list with all the possible ARIMA permutations.
perms = list(itertools.product(range(0, 11, 1), repeat=3))

# Making a list of all the possible Solvers
solver_list = ['lbfgs', 'bfgs', 'newton', 'nm', 'cg', 'ncg', 'powell']


def dataset_test_training_split(X_values, y_values, prop_training=0.65):
    assert prop_training < 1
    X_train, X_test, y_train, y_test = \
        train_test_split(X_values, y_values, test_size=1 - prop_training, train_size=prop_training, shuffle=False)
    return X_train, X_test, y_train, y_test


def evaluate_arima_model(arima_order, arima_solver):
    global X_train, X_test, y_train, y_test
    model_predictions = []
    history = [x for x in y_train]
    for t in range(len(X_test)):
        model = ARIMA(endog=history, order=arima_order)
        model_fit = model.fit(disp=False, solver=arima_solver)
        model_predictions.append(model_fit.forecast()[0])
        history = np.append(history, y_test[t])
    error = mean_squared_error(y_test, model_predictions)
    return error


def optimise_arima_model(list_of_solvers, list_of_perms):
    global X_train, X_test, y_train, y_test
    results = []
    for solver in list_of_solvers:
        for permutation in list_of_perms:
            try:
                error = evaluate_arima_model(permutation, solver)
                results.append((solver, permutation, error))
            except:
                continue

    ARIMA_opt = min(results, key=operator.itemgetter(2))
    print('ARIMA Model Optimsied')
    print('Solver: {}'.format(ARIMA_opt[0]))
    print('Order: {}'.format([1]))
    # Return Solver, ARIMA order
    return ARIMA_opt[0], ARIMA_opt[1]


# load dataset
series = read_csv('airline_passengers.csv',index_col='Month')
X_train, X_test, y_train, y_test=dataset_test_training_split(X_values=series.index,y_values=series.values)
optimise_arima_model(solver_list,perms)

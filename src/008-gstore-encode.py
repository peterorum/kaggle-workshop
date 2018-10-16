import os
import json
import sys
import csv
from pprint import pprint
import lightgbm as lgb
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from sklearn.preprocessing import LabelEncoder

pd.options.display.float_format = '{:.4f}'.format
pd.set_option('display.max_columns', None)

# https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields


def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

    df = pd.read_csv(csv_path,
                     converters={column: json.loads for column in JSON_COLUMNS},
                     dtype={'fullVisitorId': 'str'},  # to make unique
                     nrows=nrows)

    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)

    # print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")

    return df


def evaluate():

    model = lgb.LGBMRegressor(nthread=4, n_jobs=-1, verbose=-1)

    test_predictions = np.zeros(test.shape[0])
    best_score = 0

    for fold_n, (train_index, test_index) in enumerate(folds.split(x_train)):
        X_train, X_valid = x_train.iloc[train_index], x_train.iloc[test_index]
        Y_train, Y_valid = y_train.iloc[train_index], y_train.iloc[test_index]

        result = model.fit(X_train, Y_train,
                           eval_set=[(X_train, Y_train), (X_valid, Y_valid)],
                           eval_metric='rmse',
                           verbose=False, early_stopping_rounds=stop_rounds)

        # pprint(dir(model))
        best_score += model.best_score_['valid_1']['rmse']

        test_prediction = model.predict(x_test, num_iteration=model.best_iteration_)

        test_predictions += test_prediction

    test_predictions /= n_folds
    best_score /= n_folds

    return test_predictions, best_score


def objective(params):

    global iteration, best_score
    iteration += 1

    start = timer()

    _, score = evaluate(params)

    run_time = timer() - start

    # save results
    of_connection = open(results_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([iteration, score, run_time, params])
    of_connection.close()

    # save trials for resumption
    # with open('trials.json', 'w') as f:
    #     # might be trials_dict to be saved
    #     f.write(json.dumps(trials))

    best_score = min(best_score, score)

    print(f'iteration {iteration}, score {score}, best {best_score}, timer {run_time}')

    # score must be to minimize

    return {'loss': score, 'params': params, 'iteration': iteration,
            'train_time': run_time, 'status': STATUS_OK}


# add dates
def expand_dates(data, column):
    # convert & expand yyyymmdd

    data[column] = data[column].astype(str)
    data[column] = data[column].apply(lambda x: x[:4] + "-" + x[4:6] + "-" + x[6:])
    data[column] = pd.to_datetime(data[column])
    data["year"] = data[column].dt.year
    data["month"] = data[column].dt.month
    data["day"] = data[column].dt.day
    data["weekday"] = data[column].dt.weekday
    data['weekofyear'] = data[column].dt.weekofyear

    data.drop(column, axis=1, inplace=True)

    return data

#-------- main

start_time = time()

# load data

if use_sample:
    train = pd.read_csv('../input/train-sample.csv')
    train['fullVisitorId'] = train['fullVisitorId'].astype(str)
    test = pd.read_csv('../input/test-sample.csv')
    test['fullVisitorId'] = test['fullVisitorId'].astype(str)
else:
    train = load_df(f'../input/train.csv{zipext}')
    test = load_df(f'../input/test.csv{zipext}')

test['trafficSource.campaignCode'] = 0

print(f'Load data {((time() - start_time) / 60):.0f} mins')

key = 'fullVisitorId'
target = 'totals.transactionRevenue'

train[target] = train[target].fillna(0)

train = expand_dates(train, 'date')
test = expand_dates(test, 'date')

# specific conversions required
int_cols = ['totals.hits', 'totals.visits']

float_cols = [
    'totals.transactionRevenue',
]

for col in int_cols:
    train[col] = train[col].astype(int)
    test[col] = test[col].astype(int)

for col in float_cols:
    train[col] = train[col].astype(float)

    if col != target:
        test[col] = test[col].astype(float)

train[target] = train[target].fillna(0)

# evaluation is log
train_targets = np.log1p(train[target])

#----------

all_numeric_cols = [col for col in train.columns
                    if (train[col].dtype == 'int64') | (train[col].dtype == 'float64')]

categorical_cols = [col for col in train.columns if train[col].dtype == 'object']
categorical_cols.remove(key)

# replace missing numericals with mean
for col in all_numeric_cols:
    if train[col].isna().any():
        mean = train[col].mean()

        train[col].fillna(mean, inplace=True)

        if col in test.columns:
            test[col].fillna(mean, inplace=True)

# replace missing categoricals with mode
for col in categorical_cols:
    if train[col].isna().any():
        mode = train[col].mode()[0]

        train[col].fillna(mode, inplace=True)

        if col in test.columns:
            test[col].fillna(mode, inplace=True)

# feature selection via variance
train_numeric = train[all_numeric_cols].fillna(0)
select_features = VarianceThreshold(threshold=0.2)
select_features.fit(train_numeric)
numeric_cols = train_numeric.columns[select_features.get_support(indices=True)].tolist()

# remove cols without variance
for col in all_numeric_cols:
    if col not in numeric_cols:
        train.drop(col, axis=1, inplace=True)

        if col in test.columns:
            test.drop(col, axis=1, inplace=True)


# encode categoricals so all numeric

# drop if too many values, encode if few
# one-hot-encode up to 100 categories, else label encode
max_categories = train.shape[0] * 0.5
max_ohe_categories = 100

too_many_value_categorical_cols = [col for col in categorical_cols
                                   if train[col].nunique() >= max_categories]

train = train.drop(too_many_value_categorical_cols, axis=1)
test.drop([col for col in too_many_value_categorical_cols if col in test.columns], axis=1, inplace=True)

categorical_cols = np.setdiff1d(categorical_cols, too_many_value_categorical_cols)

ohe_categorical_cols = [col for col in categorical_cols
                        if train[col].nunique() <= max_ohe_categories]

label_encode_categorical_cols = [col for col in categorical_cols
                                 if (train[col].nunique() > max_ohe_categories) & (train[col].nunique() < max_categories)]

# label encode
for col in label_encode_categorical_cols:
    lbl = LabelEncoder()
    lbl.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
    train[col] = lbl.transform(list(train[col].values.astype('str')))
    test[col] = lbl.transform(list(test[col].values.astype('str')))

# one-hot encode
train = pd.get_dummies(train, columns=ohe_categorical_cols)
test = pd.get_dummies(test, columns=ohe_categorical_cols)
train, test = train.align(test, join='inner', axis=1)

# reformat col names
train.columns = [col.replace(' ', '_') for col in train.columns.tolist()]
test.columns = [col.replace(' ', '_') for col in test.columns.tolist()]

# kfolds
folds = KFold(n_splits=n_folds, shuffle=True, random_state=1)

x_train = train.drop([key], axis=1)
y_train = train_targets
x_test = test[x_train.columns]

# Create the dataset
train_set = lgb.Dataset(x_train, y_train)

# Define the search space
space = {
    'class_weight': hp.choice('class_weight', [None, 'balanced']),
    'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0)
}

if optimize:
    of_connection = open(results_file, 'w')
    writer = csv.writer(of_connection)
    writer.writerow(['iteration', 'score', 'run_time', 'params'])
    of_connection.close()

    best = fmin(fn=objective, space=space, algo=tpe.suggest,
                max_evals=max_evals, trials=trials)

    print('best', best)

    # pprint(trials.results)
    trials_dict = sorted(trials.results, key=lambda x: x['loss'])
    print(f'score {trials_dict[:1][0]["loss"]}')

else:
    params = {'class_weight': None, 'colsample_bytree': 0.9571576967509944, 'learning_rate': 0.0740494277841095, 'min_child_samples': 150,
              'num_leaves': 107, 'reg_alpha': 0.6292694233464933, 'reg_lambda': 0.7740308128390287, 'subsample_for_bin': 160000}

    test_predictions, score = evaluate(params)

    print('score', score)

    test['predicted'] = test_predictions

    grouped_test = test[['fullVisitorId', 'predicted']].groupby('fullVisitorId').sum().reset_index()
    grouped_test['PredictedLogRevenue'] = grouped_test['predicted'].apply(lambda x: 0.0 if x < 0 else x)
    grouped_test[['fullVisitorId', 'PredictedLogRevenue']].to_csv('submission.csv', index=False)

print(f'{((time() - start_time) / 60):.0f} mins\a')

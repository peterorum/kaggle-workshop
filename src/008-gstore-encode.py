import json
import lightgbm as lgb
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

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
        column_as_df.columns = ["{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)

    # print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")

    return df


def evaluate():

    model = lgb.LGBMRegressor(nthread=4, n_jobs=-1, verbose=-1)

    model.fit(x_train, y_train,
              eval_metric='rmse')

    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)

    return train_predictions, test_predictions


# load data
# delete nrows = 10000 for full, but slow, processing
train = load_df('../input/train.csv')
test = load_df('../input/test.csv')

# train.drop('trafficSource.campaignCode', inplace=True, axis=1)

key = 'fullVisitorId'
target = 'totals.transactionRevenue'

train[target] = train[target].fillna(0).astype(float)

# evaluation is log
train_targets = np.log1p(train[target])

# selected columns

train = train[[key, target, 'visitNumber', 'device.operatingSystem', 'device.deviceCategory']]

print(train.head())

# label encode a column
lbl = LabelEncoder()
col = 'device.operatingSystem'
lbl.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
train[col] = lbl.transform(list(train[col].values.astype('str')))
test[col] = lbl.transform(list(test[col].values.astype('str')))

# one-hot encode selected column
cols = ['device.deviceCategory']
train = pd.get_dummies(train, columns=cols)
test = pd.get_dummies(test, columns=cols)
train, test = train.align(test, join='inner', axis=1)

print(train.head())

x_train = train.drop([key], axis=1)
y_train = train_targets
x_test = test[x_train.columns]

train_predictions, test_predictions = evaluate()

score = np.sqrt(mean_squared_error(train_targets, train_predictions))
print('score', score)

test['predicted'] = test_predictions

grouped_test = test[['fullVisitorId', 'predicted']].groupby('fullVisitorId').sum().reset_index()
grouped_test['PredictedLogRevenue'] = grouped_test['predicted'].apply(lambda x: 0.0 if x < 0 else x)
grouped_test[['fullVisitorId', 'PredictedLogRevenue']].to_csv('submission.csv', index=False)

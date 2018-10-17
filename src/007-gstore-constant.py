import os
import json
import sys
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
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
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)

    # print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")

    return df


#-------- main

# load data

# remove nrows = 10000 for full submission

train = load_df('../input/train.csv', nrows=10000)
test = load_df('../input/test.csv', nrows=10000)

key = 'fullVisitorId'
target = 'totals.transactionRevenue'

train[target] = train[target].fillna(0).astype(float)

# use mean as prediction
result = train[target].mean()
train['predicted'] = result

print(result)

score = np.sqrt(mean_squared_error(np.log1p(train[target]), np.log1p(train.predicted)))
print('score', score)

# set all to result
test[target] = result

grouped_test = test[['fullVisitorId', target]].groupby('fullVisitorId').sum().reset_index()
grouped_test['PredictedLogRevenue'] = np.log1p(grouped_test[target])

grouped_test[['fullVisitorId', 'PredictedLogRevenue']].to_csv('submission.csv', index=False)

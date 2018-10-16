import numpy as np
import pandas as pd

# load data
train = pd.read_csv(f'../input/train.csv')
test = pd.read_csv(f'../input/test.csv')

target = 'SalePrice'

result = 200000

train['predicted'] = result

test[target] = result

predictions = test[['Id', target]]

predictions.to_csv('submission.csv', index=False)

import pandas as pd

# load data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv("../input/test.csv")

test['Survived'] = 0

predictions = test[['PassengerId', 'Survived']]

predictions.to_csv('submission.csv', index=False)

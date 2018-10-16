import pandas as pd

# load data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv("../input/test.csv")

test['Survived'] = test.Sex.apply(lambda x: 1 if x == 'female' else 0)

print(test.head())
print(test.describe())

predictions = test[['PassengerId', 'Survived']]

predictions.to_csv('submission.csv', index=False)

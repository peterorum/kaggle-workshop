import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# load data
train = pd.read_csv(f'../input/train.csv')
test = pd.read_csv(f'../input/test.csv')

target = 'SalePrice'

train = train[['LotArea', target]]

train = train.fillna(0)

x_train = train.drop(target, axis=1)
y_train = train[target]

model = lgb.LGBMRegressor()
model.fit(x_train, y_train)

train['predicted'] = model.predict(x_train)

train['predicted'] = train['predicted'].apply(lambda x: x if x >= 0 else 0)

score = np.sqrt(mean_squared_error(np.log1p(train[target]), np.log1p(train.predicted)))
print('score', score)

x_test = test[x_train.columns]
x_test = x_test.fillna(0)

predicted = model.predict(x_test)

submission = pd.DataFrame({
    "ID": test.Id,
    "SalePrice": predicted
})

submission.to_csv('submission.csv', index=False)

import xgboost as xgb
import numpy as np
import pandas as pd

# 結局 sklearn は使うんですけどもね。
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 読み込むデータを datasets の irisから 自分で指定した CSVファイルに変更
file_pass = '/Users/tomohisaonose/Python/MachineLearning/Kmeans/Values.csv'
iris_csv = pd.read_csv(file_pass, header=None)


# iloc[1:]指定しているのはヘッダを読まないため。
x = iris_csv.iloc[1:, :3]
x = x.values.astype(float)
# 確認 :print('Feature data :', X)

y = iris_csv.iloc[1:, 3]
y = y.values.astype(int)
y = y - 1
# 確認 :print('answer :', y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=42, stratify=y)

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)


xgb_params = {
    'objective': 'multi:softmax',
    'num_class': 3,
    'eval_metric': 'mlogloss'
}

evals = [(dtrain, 'train'), (dtest, 'eval')]
evals_result = {}
bst = xgb.train(xgb_params,
                dtrain,
                num_boost_round=100,
                early_stopping_rounds=10,
                evals=evals,
                evals_result=evals_result
                )


y_pred = bst.predict(dtest)
acc = accuracy_score(y_test, y_pred)
print("Accuracy : ", acc)

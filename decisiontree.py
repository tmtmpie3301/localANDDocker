# ライブラリの読み込み
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split

# irisデータの読み込み
iris = datasets.load_iris()

# 特徴量とターゲットの取得
data = iris['data']
target = iris['target']

# 学習データをテストデータを分割
train_data, test_data, train_target, test_target = train_test_split(
    data, target, test_size=0.5)

# モデル学習
model = DecisionTreeClassifier(criterion='gini')
model.fit(train_data, train_target)

# 正解率を表示
print(model.score(test_data, test_target))

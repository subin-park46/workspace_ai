from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt


# 1. 데이터 준비
iris = load_iris()
x = iris.data
y = iris.target

# 2. 데이터 분할
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=1)

# 3. 준비
model = KMeans(n_clusters=3) # iris에 setosa, versicolor, virginica가 있고, 이 3개 중에 하나로 분류해달라.

# 4. 학습
model.fit(train_x)

# 5. 예측
pred = model.predict(test_x)
for i in range(len(test_x)):
    print(f'{test_x[1]} 예측 : {iris.target_names[pred[i]]} / 실제 : {iris.target_names[test_y[i]]}')

# 어떻게 군집이 되었는지 그래프
df = pd.DataFrame(test_x)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df['category'] = pd.DataFrame(pred)

centers = pd.DataFrame(model.cluster_centers_)
centers.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
center_x = centers['sepal_length']
center_y = centers['sepal_width']

plt.scatter(df['sepal_length'], df['sepal_width'], c = df['category'])
plt.scatter(center_x, center_y, s = 100, c = 'r', marker = '*')
plt.show()
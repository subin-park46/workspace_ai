from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# 1. 데이터 준비
iris = load_iris()

# print(type(iris.data))
print(iris.feature_names)
x = iris.data

# print(iris.target)
# print(iris.target_names)
y = iris.target

"""
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
df = pd.DataFrame(x)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df['category'] = pd.DataFrame(iris.target_names[y].reshape(-1))

groups = df.groupby('category')
fig, ax = plt.subplots()
for name, group in groups:
    ax.scatter(group.sepal_length, group.sepal_width, marker='.', label=name)

ax.legend()
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.show()
"""

# 2. 데이터 분할
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=1)

# 3. 준비
model = LogisticRegression()

# 4. 학습
model.fit(train_x, train_y)

# 5. 예측
pred = model.predict(test_x)
for i in range(len(test_x)):
    print(f'{test_x[1]} 예측 : {iris.target_names[pred[i]]} / 실제 : {iris.target_names[test_y[i]]}')
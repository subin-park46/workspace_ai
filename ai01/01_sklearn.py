from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# plt.plot(x, y)
# plt.show()

model = LinearRegression() # 선형회귀
model.fit(x.reshape(-1, 1), y) # 형태 바꾸어 학습.

test_x = [[6], [7], [8], [9], [10]]
predict = model.predict(test_x)
print(predict)
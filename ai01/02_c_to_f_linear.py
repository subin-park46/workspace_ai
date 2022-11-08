import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def celsius_to_fahrenheit(x):
    return x * 1.8 + 32


# 1. data 준비
x = np.array(range(0, 10))
y = celsius_to_fahrenheit(x)

# 2. data 분할
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=1) # random_state : 호출할 때마다 동일한 학습/테스트용 데이터 세트를 생성하기 위해 주어지는 난수 값

# 3. 준비
model = LinearRegression()

# 4. 학습
model.fit(train_x.reshape(-1, 1), train_y)

# 5. 예측
predict = model.predict(test_x.reshape(-1, 1))
print(predict)

print(f"30'c -> {model.predict([[30]])}'f")

# 6. 평가
accuracy = model.score(test_x.reshape(-1, 1), test_y)
print(accuracy)
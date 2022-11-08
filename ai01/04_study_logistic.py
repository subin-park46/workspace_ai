from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np


# 1. 데이터 준비
# [공부시간, 과외시간] ->
x = [
    [1, 0],
    [2, 0],
    [5, 1],
    [2, 3],
    [3, 3],
    [8, 1],
    [10, 0]
]

# 시험에 pass[1] / fail[0]
y = [
    [0],
    [0],
    [0],
    [0],
    [1],
    [1],
    [1]
]

# 2. 데이터 분할
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=1)

# 3. 준비
model = LogisticRegression()

# 4. 학습
model.fit(train_x, np.ravel(train_y)) # np.ravel은 다차원을 1차원으로 변환.

# 5. 예측
pred = model.predict(test_x)

for i in range(len(test_x)):
    print(f'{test_x[i][0]} 시간 공부하고 {test_x[i][1]} 시간 과외 받으면 : {"pass" if pred[i] ==  1 else "fail"}')

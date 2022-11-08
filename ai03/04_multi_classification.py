from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import adam_v2
import numpy as np


# 데이터 준비
# 4번의 쪽지 시험
data_x = [
    [10, 7, 8, 5],
    [8, 8, 9, 4],
    [7, 8, 2, 3],
    [6, 3, 9, 3],
    [7, 5, 7, 4],
    [3, 5, 6, 2],
    [2, 4, 3, 1]
]

# 상 | 중 | 하 -> one hot encoding
data_y = [
    [1, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 1]
]

# 사용할 것
model = Sequential()

# 준비
model.add(Dense(units=10, input_dim=4))
model.add(Dense(units=30, input_dim=10))
model.add(Dense(units=20, input_dim=30))
model.add(Dense(units=3, input_dim=20, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=adam_v2.Adam(learning_rate=0.1))

# 학습
model.fit(data_x, data_y, epochs=100, verbose=1)

# 예측
print(model.predict([[4, 9, 5, 4]]))
# print(list(map(lambda x: np.argmax(x.round(2)), model.predict([[4, 9, 5, 4]]))))
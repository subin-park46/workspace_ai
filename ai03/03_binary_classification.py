from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import adam_v2


# 데이터 준비
# [공부시간, 과외시간] ->
data_x = [
    [1, 0],
    [2, 0],
    [5, 1],
    [2, 3],
    [3, 3],
    [8, 1],
    [10, 0]
]

# 시험에 pass[1] / fail[0]
data_y = [
    [0],
    [0],
    [0],
    [0],
    [1],
    [1],
    [1]
]

# 사용할 것
model = Sequential()

# 준비
model.add(Dense(units=10, input_dim=2, activation='relu'))
model.add(Dense(units=40, input_dim=10))
model.add(Dense(10))
model.add(Dense(units=1, input_dim=10, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=adam_v2.Adam(learning_rate=0.1))

# 학습
model.fit(data_x, data_y, epochs=50, verbose=1)

# 예측
print(model.predict([[5, 3]]))
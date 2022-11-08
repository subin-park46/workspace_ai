from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import adam_v2


# 데이터 준비
# 세 번의 쪽지시험 결과를 가지고 실제 시험 점수를 예측하고 싶다.
data_x = [
    [73, 80, 75],
    [93, 88, 93],
    [89, 91, 90],
    [96, 89, 100],
    [73, 66, 70]
]

data_y = [
    [80],
    [91],
    [88],
    [94],
    [61]
]

# 사용할 것
model = Sequential()

# 준비
model.add(Dense(units=10, input_dim=3))
model.add(Dense(units=20))
model.add(Dense(10))
model.add(Dense(units=1))

model.compile(loss='mse', optimizer=adam_v2.Adam(learning_rate=0.1))

# 학습
model.fit(data_x, data_y, epochs=100, verbose=1)

# 학습
test_x = [[100, 80, 78]]
print(model.predict(test_x))
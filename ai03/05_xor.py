from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import adam_v2


# 데이터 준비
x = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

y = [
    [0],
    [1],
    [1],
    [0]
]

# 사용할 것
model = Sequential()

# 준비
model.add(Dense(units=10, input_dim=2, activation='sigmoid'))
model.add(Dense(units=20, input_dim=10, activation='sigmoid'))
model.add(Dense(units=10, input_dim=20, activation='sigmoid'))
model.add(Dense(units=1, input_dim=10, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=adam_v2.Adam(learning_rate=0.1))

# 학습
model.fit(x, y, epochs=100, verbose=1)

# 예측
print(model.predict(x).round())
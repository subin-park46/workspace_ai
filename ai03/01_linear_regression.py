import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import adam_v2


# 데이터 준비
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

# 데이터 분할
model = Sequential() # 레이어를 선형으로 연결

# 준비
model.add(Dense(units=10, input_dim=1)) # x = 1개 들어오고, 10개 나간다
model.add(Dense(units=40, input_dim=10))
model.add(Dense(units=20, input_dim=40))
model.add(Dense(units=1, input_dim=20)) # Linear 때문에 최종 나가는 값 1개

model.compile(loss='mse', optimizer=adam_v2.Adam(learning_rate=0.1))

# 학습
model.fit(x, y, epochs=100, verbose=1) # 학습 진행 과정 T(1) / F(0)

# 예측
test_x = [1, 3, 5, 7, 9]
predict = model.predict(test_x)
print(predict)
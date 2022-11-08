import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

# 1. data 준비
df = pd.read_csv("./weight_height.csv", encoding='euc-kr')

# 원하는 데이터만 사용하기 위해 전처리
df = df[["학교명", "학년", "성별", "키", "몸무게"]]
df.dropna(inplace=True)

# df['학교명'] -> '초등학교' : 0 / '중학교' : 6 / '고등학교' : 9 / + df['학년'] => df['grade']
df['grade'] = list(map(lambda x: 0 if x[-4:] == '초등학교' else 6 if x[-3:] == '중학교' else 9, df['학교명'])) + df['학년']

df.drop(['학교명', '학년'], axis='columns', inplace=True)

df.columns = ['gender', 'height', 'weight', 'grade']

# 남자 : 0 / 여자 : 1
df['gender'] = df['gender'].map(lambda x: 0 if x == "남" else 1)

x = df[["weight", "gender"]]
y = df["height"]

# 다항 회귀 (Polynornial features) : 특성이 두개니까, 특성을 곱해서 다차원으로 바꾸자.
# [1, a, b, a^2, ab, b^2]
ploy = PolynomialFeatures()
x = ploy.fit(x).transform(x)

# 2. data 분할
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=1)
# train_x = train_x.values.reshape(-1, 1)
# test_x = test_x.values.reshape(-1, 1)

# 3. 준비
model = LinearRegression()

# 4. 학습
model.fit(train_x, train_y)

# 5. 예측
boy = ploy.fit([[80, 0]]).transform([[80, 0]])
print(f'몸무게가 80인 남학생의 키 예측 : {model.predict(boy)}')

# 6. 평가
accuracy = model.score(test_x, test_y)
print(accuracy)

predict = model.predict(test_x)

plt.plot(test_x, test_y, 'b.')
plt.plot(test_x, predict, 'r.')
plt.show()
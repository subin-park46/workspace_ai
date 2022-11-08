import tensorflow as tf


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

# [None, 숫자] = 행은 상관 없고, 뒤의 숫자가 데이터 갯수
x = tf.placeholder(shape=[None, 3], dtype=tf.float32)
y = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 데이터 분할
w = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

h = tf.matmul(x, w) + b # matmul : 행렬 곱

# 준비
loss = tf.reduce_mean(tf.square(h - y))

learning_rate = 0.00004
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습
epochs = 10000
for step in range(epochs):
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x: data_x, y: data_y})
    if step % 1000 == 0:
        print(f"w : {w_val} \t b: {b_val} \t loss : {loss_val}")

# 예측
print(sess.run(h, feed_dict={x: [[100, 80, 87]]}))
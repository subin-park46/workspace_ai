import tensorflow as tf


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

x = tf.placeholder(shape=[None, 4], dtype=tf.float32)
y = tf.placeholder(shape=[None, 3], dtype=tf.float32)

# 데이터 분할
w = tf.Variable(tf.random_normal([4, 3])) # 3은 상중하 데이터 값 3개라서
b = tf.Variable(tf.random_normal([3]))

logits = tf.matmul(x, w) + b
h = tf.nn.softmax(logits)

# 준비
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습
for step in range(3000):
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x: data_x, y: data_y})
    if step % 300 == 0:
        print(f'w : {w_val} \t b: {b_val} \t loss : {loss_val}')

# 예측
print(sess.run(h, feed_dict={x: [[3, 9, 5, 5]]}))
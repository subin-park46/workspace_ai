import tensorflow as tf


# 데이터 준비
data_x = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

data_y = [
    [0],
    [1],
    [1],
    [0]
]

x = tf.placeholder(shape=[None, 2], dtype=tf.float32)
y = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 데이터 분할
w = tf.Variable(tf.random_normal([2, 1]))
b = tf.Variable(tf.random_normal([1]))

logits = tf.matmul(x, w) + b
h = tf.nn.softmax(logits)

# 준비
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습
for step in range(10000):
    _, loss_val = sess.run([train, loss], feed_dict={x: data_x, y: data_y})
    if step % 1000 == 0:
        print(f'loss : {loss_val}')

# 예측
print(sess.run(h, feed_dict={x: data_x}))
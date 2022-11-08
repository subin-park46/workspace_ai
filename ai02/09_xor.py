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
# input layer
w1 = tf.Variable(tf.random_normal([2, 10]))
b1 = tf.Variable(tf.random_normal([10]))
layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)

# hidden layer
w2 = tf.Variable(tf.random_normal([10, 20]))
b2 = tf.Variable(tf.random_normal([20]))
layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2) # layer1 값과 w2를 곱한 후, b2를 더함.

w3 = tf.Variable(tf.random_normal([20, 10]))
b3 = tf.Variable(tf.random_normal([10]))
layer3 = tf.sigmoid(tf.matmul(layer2, w3) + b3) # layer2 값과 w3을 곱한 후, b3을 더함.

# output layer
w4 = tf.Variable(tf.random_normal([10, 1]))
b4 = tf.Variable(tf.random_normal([1]))
logits = tf.matmul(layer3, w4) + b4
h = tf.sigmoid(logits)

# 준비
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))

train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습
for step in range(10000):
    _, loss_val = sess.run([train, loss], feed_dict={x: data_x, y: data_y})
    if step % 1000 == 0:
        print(f'loss : {loss_val}')

# 예측
print(sess.run(h, feed_dict={x: data_x}))
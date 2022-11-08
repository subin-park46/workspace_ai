import tensorflow as tf


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

x = tf.placeholder(shape=[None, 2], dtype=tf.float32)
y = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 데이터 분할
w = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

logit = tf.matmul(x, w) + b

# sigmoid : 0 ~ 1 사이의 실수 (h > 0.5 : True)
h = tf.sigmoid(logit)

# 준비
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=y))

learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습
epochs = 10000
for step in range(epochs):
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict = {x: data_x, y: data_y})
    if step % 500 == 0:
        print(f'w : {w_val} \t b: {b_val} \t ;pass : {loss_val}')

# 예측
print(f'4시간 공부하고 4시간 과외 받으면 : {"pass" if (sess.run(h, feed_dict={x: [[4, 4]]})) > 0.5 else "fail"}')
import tensorflow as tf


# 데이터 준비
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# 가설 설정
# H = w * x + b
w = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

h = w * x + b

# 준비
loss = tf.reduce_mean(tf.square(h - y))

# 경사하강법 (gradient descent)
# learning_rate : 얼만큼씩 움직일건지 (0.01)
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss) # 모델 학습 시 손실 값을 최소화

# session
sess = tf.Session()

# 변수 초기화
sess.run(tf.global_variables_initializer())

# 학습
epochs = 5000
for step in range(epochs):
    tmp, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x: [1, 2, 3, 4, 5], y: [3, 5, 7, 9, 11]})
    if step % 100 == 0:
        print(f'w:{w_val} \t b:{b_val} \t loss: {loss_val}')

# 예측
print(sess.run(h, feed_dict={x: [10, 11, 12, 13, 14]}))

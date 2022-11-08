import tensorflow as tf


# placeholder : 그래프르 실행하는 시점에 데이터를 입력 받아서 실행.
node1 = tf.placeholder(dtype=tf.float32)
node2 = tf.placeholder(dtype=tf.float32)
node3 = node1 + node2
sess = tf.Session()

x = [10, 20, 30]
y = [40, 50, 60]

print(sess.run(node3, feed_dict={node1: x, node2: y}))
import tensorflow.compat.v1 as tf

# np.float32 == tf.float32
node1 = tf.constant(10, dtype=tf.float32)
node2 = tf.constant(20, dtype=tf.float32)

node3 = node1 + node2

sess = tf.Session()
print(sess.run(node3))

# node1과 node3을 같이 실행
print(sess.run([node1, node3]))
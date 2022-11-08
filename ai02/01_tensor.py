import tensorflow as tf


# 상수 노드
node = tf.constant(100)

# 그래프 실행 시켜주는 역할 (runner)
sess = tf.Session()

# 노드 실행
print(sess.run(node))

"""
Tensor : 데이터 저장 객체 (placeholder)
Variable : weight, bias
Operation : H = w * x + b (수식) -> graph : tensor -> operation -> tensor -> operation
Session : 실행환경 (학습)
"""
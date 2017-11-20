import tensorflow as tf

matrix_1 = tf.constant([[3, 3]])  # create matrix of 2 cols and 1 row
matrix_2 = tf.constant([[2], [2]])  # create matrix of 1 col and 2 rows

product = tf.matmul(matrix_1, matrix_2)  # ==> np.dot(m1, m2)

# method 1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

# method 2
with tf.Session() as sess:
    result = sess.run(product)
    print(result)



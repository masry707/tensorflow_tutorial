import tensorflow as tf

# placeholders are prettu much the same as the variables but the only diff
# is you can assign there values later in the sess.run()

input_1 = tf.placeholder(tf.float32)
input_2 = tf.placeholder(tf.float32)
output = tf.multiply(input_1, input_2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input_1: [7.], input_2: [3.]}))

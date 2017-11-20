import tensorflow as tf
import numpy as np


# create the inputs
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3


# create the structure of the NN
weights = tf.Variable(tf.random_uniform([1], -1, 1))
biases = tf.Variable(tf.random_uniform([1], -1, 1))

# the problem we solve
y = x_data * weights + biases


loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train_step = optimizer.minimize(loss=loss)

# very very important
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# how the train works
for i in range(201):
    sess.run(train_step)
    if i % 20 == 0:
        print(i, sess.run(weights), sess.run(biases))


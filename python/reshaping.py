import tensorflow as tf

a = tf.placeholder(tf.float32, [None] + [None] + [10])
shape = a.get_shape()
b = tf.reshape(a, [None] + [10])



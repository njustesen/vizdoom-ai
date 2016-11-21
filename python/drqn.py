#!/usr/bin/env python
# -*- coding: utf-8 -*-
from random import sample, randint, random
import numpy as np
import tensorflow as tf

resolution = (64, 32)
available_actions_count = 4
batch_size = 32
learning_rate = 0.00025

# Start TF session
print("Starting session")
session = tf.Session()

print("Creating model")

# Input - [batch_size, time, x, y, channels]
s1_ = tf.placeholder(tf.float32, [None, None, resolution[0], resolution[1], 1], name="State")

# Number of frames in input
seq_length_ = tf.placeholder(tf.int32)

# Batch size
batch_size_ = tf.placeholder(tf.int32)

# [batch_size, time, actions]
target_q_ = tf.placeholder(tf.float32, [None, None, available_actions_count], name="TargetQ")

# Reshape [batch_size * time, x, y, channels]
s1_reshaped = tf.reshape(tensor=s1_, shape=[-1, resolution[0], resolution[1], 1])

# 2 convolutional layers with ReLu activation
conv1 = tf.contrib.layers.convolution2d(s1_reshaped, num_outputs=32, kernel_size=[6, 6], stride=[3, 3],
                                activation_fn=tf.nn.relu,
                                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                biases_initializer=tf.constant_initializer(0.1))
conv2 = tf.contrib.layers.convolution2d(conv1, num_outputs=16, kernel_size=[3, 3], stride=[2, 2],
                                activation_fn=tf.nn.relu,
                                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                biases_initializer=tf.constant_initializer(0.1))

# Flatten
conv2_flat = tf.contrib.layers.flatten(conv2)

# Fully connected layer [batch_size * time, num_nodes]
fc1 = tf.contrib.layers.fully_connected(conv2_flat, num_outputs=32, activation_fn=tf.nn.relu,
                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                biases_initializer=tf.constant_initializer(0.1))

# [batch_size, time, n_dense]
fc1_reshaped = tf.reshape(fc1, shape=[-1, seq_length_, 32])

# Transpose to [time, batch_size, n_dense]
fc1_transposed = tf.transpose(fc1_reshaped, [1, 0, 2])

# RNN
num_units = 512
cell = tf.nn.rnn_cell.GRUCell(num_units)
zero_state = cell.zero_state(batch_size_, tf.float32)

# Initial RNN state
rnn_state_ = tf.placeholder_with_default(zero_state, [None, num_units])

rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, fc1_transposed, initial_state=rnn_state_, time_major=True)

# Transpose to [batch_size, time, n_dense]
rnn_transposed = tf.transpose(rnn_outputs, [1, 0, 2])

# Output
q = tf.contrib.layers.fully_connected(rnn_transposed,
                                      num_outputs=available_actions_count,
                                      activation_fn=None,
                                      weights_initializer=tf.contrib.layers.xavier_initializer(),
                                      biases_initializer=tf.constant_initializer(0.1))

# Best action
best_a = tf.argmax(q, 2)

# Calculate loss
loss = tf.contrib.losses.mean_squared_error(q, target_q_)

# Update the parameters according to the computed gradient using RMSProp.
optimizer = tf.train.RMSPropOptimizer(learning_rate)
train_step = optimizer.minimize(loss)


def function_learn(s1, target_q, seq_length):
    feed_dict = {s1_: s1, target_q_: target_q, seq_length_: seq_length, batch_size_:batch_size}
    l, _ = session.run([loss, train_step], feed_dict=feed_dict)
    return l


def function_get_q_values(state, seq_length):
    return session.run(q, feed_dict={s1_: state, seq_length_: seq_length, batch_size_:batch_size})


def function_get_best_action(state):
    return session.run([best_a, final_state], feed_dict={s1_: state, seq_length_: 1, batch_size_:1})

def function_get_best_action_rnn_state(state, rnn_init_state):
    return session.run([best_a, final_state], feed_dict={s1_: state, rnn_state_: rnn_init_state, seq_length_: 1, batch_size_:1})



# Initialize
init = tf.initialize_all_variables()
session.run(init)

# -- Learn --
# Get batch_size sequences of <s1, s2, r, a> from replay memory
max_time = 8
s1 = np.zeros((batch_size, max_time, resolution[0], resolution[1], 1))
target_q = function_get_q_values(s1, max_time)
# calculate target_q using the q-learning update rule with s2, r, a
l = function_learn(s1, target_q, max_time)
print(l)

# -- Play --
# Current frame
s1 = np.zeros((1, 1, resolution[0], resolution[1], 1))

# Get best action from s1
a, state = function_get_best_action(s1)
action = a[-1][0]
print(action)

a2, state2 = function_get_best_action_rnn_state(s1, state)
action2 = a2[-1][0]
print(action2)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from vizdoom import *
import itertools as it
from random import sample, randint, random
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
import tensorflow as tf
from tqdm import trange
import math


class ReplayMemory:
    def __init__(self, capacity, resolution):
        channels = 1
        state_shape = (capacity, resolution[0], resolution[1], channels)
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos, :, :, 0] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :, :, 0] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]


class Learner:

    def __init__(self,
                 available_actions_count,
                 learning_rate=0.00025,
                 discount_factor=0.99,
                 epochs=20,
                 learning_steps_per_epoch=2000,
                 replay_memory_size=10000,
                 batch_size=64,
                 test_episodes_per_epoch=2,
                 frame_repeat=12,
                 p_decay=0.45,
                 resolution=(30, 45),
                 model_savefile="/tmp/model.ckpt",
                 save_model=True,
                 load_model=False):

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epochs = epochs
        self.learning_steps_per_epoch = learning_steps_per_epoch
        self.replay_memory_size = replay_memory_size
        self.batch_size = batch_size
        self.test_episodes_per_epoch = test_episodes_per_epoch
        self.frame_repeat = frame_repeat
        self.p_decay = p_decay
        self.resolution = resolution
        self.available_actions_count = available_actions_count
        self.model_savefile = model_savefile
        self.save_model = save_model
        self.load_model = load_model

        # Positions traversed during an episode
        self.positions = []

        # Create replay memory which will store the transitions
        self.memory = ReplayMemory(capacity=replay_memory_size, resolution=resolution)

        # Start TF session
        self.session = tf.Session()

        # Create the input variables
        s1_ = tf.placeholder(tf.float32, [None] + list(self.resolution) + [1], name="State")
        a_ = tf.placeholder(tf.int32, [None], name="Action")
        target_q_ = tf.placeholder(tf.float32, [None, available_actions_count], name="TargetQ")

        # Add 2 convolutional layers with ReLu activation
        conv1 = tf.contrib.layers.convolution2d(s1_, num_outputs=8, kernel_size=[6, 6], stride=[3, 3],
                                        activation_fn=tf.nn.relu,
                                        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                        biases_initializer=tf.constant_initializer(0.1))
        conv2 = tf.contrib.layers.convolution2d(conv1, num_outputs=8, kernel_size=[3, 3], stride=[2, 2],
                                        activation_fn=tf.nn.relu,
                                        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                        biases_initializer=tf.constant_initializer(0.1))
        conv2_flat = tf.contrib.layers.flatten(conv2)

        #conv2_flat = tf.contrib.layers.DropoutLayer(conv2_flat, keep=0.5, name='dropout')

        fc1 = tf.contrib.layers.fully_connected(conv2_flat, num_outputs=128, activation_fn=tf.nn.relu,
                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                        biases_initializer=tf.constant_initializer(0.1))

        #fc1 = tf.contrib.layers.DropoutLayer(fc1, keep=0.5, name='dropout')

        #gru = tf.tensorlayer.RNNLayer(fc1, cell_fn=tf.nn.rnn_cell.GRUCell, n_hidden=128, n_steps=1, return_seq_2d=False)

        #gru = tf.contrib.layers.DropoutLayer(gru, keep=0.5, name='dropout')

        q = tf.contrib.layers.fully_connected(fc1, num_outputs=self.available_actions_count, activation_fn=None,
                                      weights_initializer=tf.contrib.layers.xavier_initializer(),
                                      biases_initializer=tf.constant_initializer(0.1))
        best_a = tf.argmax(q, 1)

        loss = tf.contrib.losses.mean_squared_error(q, target_q_)

        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        # Update the parameters according to the computed gradient using RMSProp.
        train_step = optimizer.minimize(loss)

        def function_learn(s1, target_q):
            feed_dict = {s1_: s1, target_q_: target_q}
            l, _ = self.session.run([loss, train_step], feed_dict=feed_dict)
            return l

        def function_get_q_values(state):
            return self.session.run(q, feed_dict={s1_: state})

        def function_get_best_action(state):
            return self.session.run(best_a, feed_dict={s1_: state})

        def function_simple_get_best_action(state):
            return function_get_best_action(state.reshape([1, self.resolution[0], self.resolution[1], 1]))[0]

        self.fn_learn = function_learn
        self.fn_get_q_values = function_get_q_values
        self.fn_get_best_action = function_simple_get_best_action

    def learn_from_memory(self):
        """ Learns from a single transition (making use of replay memory).
        s2 is ignored if s2_isterminal """

        # Get a random minibatch from the replay memory and learns from it.
        if self.memory.size > self.batch_size:
            s1, a, s2, isterminal, r = self.memory.get_sample(self.batch_size)

            q2 = np.max(self.fn_get_q_values(s2), axis=1)
            target_q = self.fn_get_q_values(s1)
            # target differs from q only for the selected action. The following means:
            # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r
            target_q[np.arange(target_q.shape[0]), a] = r + self.discount_factor * (1 - isterminal) * q2
            self.fn_learn(s1, target_q)


    def perform_learning_step(self, game, actions, epoch, reward_exploration):
        """ Makes an action according to eps-greedy policy, observes the result
        (next state, reward) and learns from the transition"""

        def exploration_rate(epoch):
            """# Define exploration rate change over time"""
            start_eps = 1.0
            end_eps = 0.1
            const_eps_epochs = 0.1 * self.epochs  # 10% of learning time
            eps_decay_epochs = 0.6 * self.epochs  # 60% of learning time

            if epoch < const_eps_epochs:
                return start_eps
            elif epoch < eps_decay_epochs:
                # Linear decay
                return start_eps - (epoch - const_eps_epochs) / \
                                   (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
            else:
                return end_eps

        s1 = self.preprocess(game.get_state().screen_buffer)

        # With probability eps make a random action.
        eps = exploration_rate(epoch)
        if random() <= eps:
            a = randint(0, len(actions) - 1)
        else:
            # Choose the best action according to the network.
            a = self.fn_get_best_action(s1)

        reward = game.make_action(actions[a], self.frame_repeat)
        if reward_exploration:
            reward = self.position_reward(game, append=True)

        isterminal = game.is_episode_finished()
        s2 = self.preprocess(game.get_state().screen_buffer) if not isterminal else None

        # Remember the transition that was just experienced.
        self.memory.add_transition(s1, a, s2, isterminal, reward)

        self.learn_from_memory()

    def get_position(self, game):
        return (game.get_game_variable(GameVariable.PLAYER_POSITION_X), game.get_game_variable(GameVariable.PLAYER_POSITION_Y))

    def position_reward(self, game, append):
        pos = self.get_position(game)
        p_reward = 0
        idx = 0
        for p in reversed(self.positions):
            distance = math.sqrt((pos[0] - p[0])**2 + (pos[1] - p[1])**2)
            p_reward += ((self.p_decay**idx)*2) * distance / 100
            idx += 1
        if append:
            self.positions.append(pos)
        return p_reward

    def preprocess(self, img):
        """ Converts and down-samples the input image. """
        img = skimage.transform.resize(img, self.resolution)
        img = img.astype(np.float32)
        return img

    def learn(self, game, actions, visual=False, reward_exploration=False):
        '''
        game.set_window_visible(visual)
        game.set_mode(Mode.PLAYER)
        game.init()
        '''
        game.set_window_visible(visual)
        game.set_mode(Mode.PLAYER)
        game.init()

        saver = tf.train.Saver()
        if self.load_model:
            print("Loading model from: ", self.model_savefile)
            saver.restore(self.session, self.model_savefile)
        else:
            init = tf.initialize_all_variables()
            self.session.run(init)

        print("Starting the training!")

        time_start = time()
        for epoch in range(self.epochs):
            print("\nEpoch %d\n-------" % (epoch + 1))
            train_episodes_finished = 0
            train_scores = []

            print("Training...")
            game.new_episode()
            self.positions = []
            score = 0
            for learning_step in trange(self.learning_steps_per_epoch):
                self.perform_learning_step(game, actions, epoch, reward_exploration)
                if reward_exploration:
                    score += self.position_reward(game, append=False)
                if game.is_episode_finished():
                    if not reward_exploration:
                        score = game.get_total_reward()
                    train_scores.append(score)
                    game.new_episode()
                    train_episodes_finished += 1
                    self.positions = []
                    score = 0

            print("%d training episodes played." % train_episodes_finished)

            train_scores = np.array(train_scores)

            print("Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()), \
                  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

            print("\nTesting...")
            test_scores = []
            for test_episode in trange(self.test_episodes_per_epoch):
                game.new_episode()
                self.positions = []
                score = 0
                while not game.is_episode_finished():
                    state = self.preprocess(game.get_state().screen_buffer)
                    best_action_index = self.fn_get_best_action(state)
                    game.make_action(actions[best_action_index], self.frame_repeat)
                    if reward_exploration:
                        score += self.position_reward(game, append=True)
                if not reward_exploration:
                    score = game.get_total_reward()
                test_scores.append(score)

            test_scores = np.array(test_scores)
            print("Results: mean: %.1f±%.1f," % (
                test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
                  "max: %.1f" % test_scores.max())

            print("Saving the network weigths to:", self.model_savefile)
            saver.save(self.session, self.model_savefile)
            # pickle.dump(get_all_param_values(net), open('weights.dump', "wb"))

            print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

        game.close()
        print("======================================")
        print("Training finished.")

    def play(self, game, actions, episodes_to_watch=1, reward_exploration=False):
        game.set_window_visible(True)
        game.set_mode(Mode.ASYNC_PLAYER)
        game.init()

        print("Loading model from: ", self.model_savefile)
        saver = tf.train.Saver()
        saver.restore(self.session, self.model_savefile)

        for _ in range(episodes_to_watch):
            game.new_episode()
            score = 0
            while not game.is_episode_finished():
                state = self.preprocess(game.get_state().screen_buffer)
                best_action_index = self.fn_get_best_action(state)
                # Instead of make_action(a, frame_repeat) in order to make the animation smooth
                game.set_action(actions[best_action_index])
                for _ in range(self.frame_repeat):
                    game.advance_action()

                if reward_exploration:
                    score += self.position_reward(game, append=True)

            # Sleep between episodes
            sleep(1.0)
            if not reward_exploration:
                score = game.get_total_reward()
            print("Total score: ", score)


# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    #game.init()
    print("Doom initialized.")
    return game

#config = "../../examples/config/rocket_basic.cfg"
#config = "../../examples/config/basic.cfg"
config = "../../examples/config/simpler_basic.cfg"
#config = "../../examples/config/my_way_home.cfg"
game = initialize_vizdoom(config)
n = game.get_available_buttons_size()
actions = [list(a) for a in it.product([0, 1], repeat=n)]
learner = Learner(available_actions_count=len(actions), frame_repeat=8, epochs=20, test_episodes_per_epoch=10)
learner.learn(game, actions, visual=True, reward_exploration=True)
learner.play(game, actions,  episodes_to_watch=20, reward_exploration=True)
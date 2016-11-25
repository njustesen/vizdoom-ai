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
import experience_replay as er
import os

class Learner:

    def __init__(self,
                 available_actions_count,
                 learning_rate=0.00025,
                 discount_factor=0.99,
                 epochs=20,
                 hidden_nodes=4608,
                 conv1_filters=32,
                 conv2_filters=64,
                 learning_steps_per_epoch=2000,
                 replay_memory_size=10000,
                 batch_size=64,
                 update_every=4,
                 test_episodes_per_epoch=2,
                 frame_repeat=12,
                 p_decay=0.95,
                 max_history=10,
                 observation_history=4,
                 reward_exploration=False,
                 resolution=(30, 45),
                 model_savefile="/tmp/model.ckpt",
                 save_model=True,
                 load_model=False):

        self.update_every = update_every
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epochs = epochs
        self.learning_steps_per_epoch = learning_steps_per_epoch
        self.replay_memory_size = replay_memory_size
        self.batch_size = batch_size
        self.test_episodes_per_epoch = test_episodes_per_epoch
        self.frame_repeat = frame_repeat
        self.p_decay = p_decay
        self.max_history = max_history
        self.observation_history = observation_history
        self.resolution = resolution
        self.available_actions_count = available_actions_count
        self.model_savefile = model_savefile
        self.save_model = save_model
        self.load_model = load_model
        self.reward_exploration = reward_exploration

        # Positions traversed during an episode
        self.positions = []

        # Create replay memory which will store the transitions
        print("Creating replay memory")
        self.memory = er.ReplayMemory(capacity=replay_memory_size, resolution=resolution)

        # Start TF session
        print("Starting session")
        self.session = tf.Session()

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
        loss = tf.contrib.losses.mean_squared_error(q[:, self.observation_history:, :], target_q_[:, self.observation_history:, :])

        # Update the parameters according to the computed gradient using RMSProp.
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        train_step = optimizer.minimize(loss)

        def function_learn(s1, target_q, seq_length):
            feed_dict = {s1_: s1, target_q_: target_q, seq_length_: seq_length, batch_size_:batch_size}
            l, _ = self.session.run([loss, train_step], feed_dict=feed_dict)
            return l

        def function_get_q_values(state, seq_length):
            return self.session.run(q, feed_dict={s1_: state, seq_length_: seq_length, batch_size_:batch_size})

        def function_get_best_action(state):
            a, s = self.session.run([best_a, final_state], feed_dict={s1_: state, seq_length_: 1, batch_size_: 1})
            return a, s

        def function_get_best_action_rnn_state(state, rnn_init_state):
            return self.session.run([best_a, final_state], feed_dict={s1_: state, rnn_state_: rnn_init_state, seq_length_: 1, batch_size_:1})

        def function_simple_get_best_action(state):
            a, s = function_get_best_action(state.reshape([1, 1, self.resolution[0], self.resolution[1], 1]))
            return a[0][0], s

        def function_simple_get_best_action_rnn_state(state, rnn_init_state):
            a, s = function_get_best_action_rnn_state(state.reshape([1, 1, self.resolution[0], self.resolution[1], 1]), rnn_init_state)
            return a[0][0], s

        self.fn_learn = function_learn
        self.fn_get_q_values = function_get_q_values
        self.fn_get_best_action = function_simple_get_best_action
        self.fn_get_best_action_rnn = function_simple_get_best_action_rnn_state

        print("Model created")

        self.rnn_state = zero_state
        self.rnn_new_state = True

    def learn_from_memory(self):
        """ Learns from a single transition (making use of replay memory).
        s2 is ignored if s2_isterminal """

        # Get a random minibatch from the replay memory and learns from it.
        if self.memory.size > self.batch_size + self.max_history:
            #s1, a, s2, isterminal, r = self.memory.get_sample(self.batch_size)
            s1, a, s2, isterminal, r = self.memory.get_sequence(self.batch_size, self.max_history)

            q2 = np.max(self.fn_get_q_values(s2, self.max_history), axis=2)
            target_q = self.fn_get_q_values(s1, self.max_history)
            # target differs from q only for the selected action. The following means:
            # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r

            # Update target q
            #target_q[np.arange(target_q.shape[0]), a] = r + self.discount_factor * (1 - isterminal) * q2

            b_idx = 0
            for batch in target_q:
                for t_idx in range(self.observation_history, len(batch)):
                    for a_idx in range(0, self.available_actions_count):
                        reward = r[b_idx][t_idx]
                        ist = isterminal[b_idx][t_idx]
                        q2_max = q2[b_idx][t_idx]
                        target_q[b_idx][t_idx][a_idx] = reward + self.discount_factor * (1 - ist) * q2_max
                b_idx += 1

            # Remove beginning of sequence - dont learn from these
            #target_q = target_q[:, self.observation_history:, :]

            self.fn_learn(s1, target_q, self.max_history)

    def exploration_rate(self, epoch, linear=False):
        """# Define exploration rate change over time"""
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.1 * self.epochs  # 10% of learning time
        eps_decay_epochs = 0.6 * self.epochs  # 60% of learning time

        if linear:
            return max(1 - (epoch / self.epochs), end_eps)

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return start_eps - (epoch - const_eps_epochs) / \
                               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps

    def perform_learning_step(self, game, actions, epoch, reward_exploration, learning_step):
        """ Makes an action according to eps-greedy policy, observes the result
        (next state, reward) and learns from the transition"""

        s1 = self.preprocess(game.get_state().screen_buffer)

        # With probability eps make a random action.
        eps = self.exploration_rate(epoch, linear=False)
        if random() <= eps:
            a = randint(0, len(actions) - 1)
        else:
            # Use last hidden state?
            if self.rnn_new_state:
                # Choose the best action according to the network.
                a, s = self.fn_get_best_action(s1)
                self.rnn_state = s
            else:
                a, s = self.fn_get_best_action_rnn(s1, self.rnn_state)
                self.rnn_state = s

        reward = game.make_action(actions[a], self.frame_repeat)
        if reward_exploration:
            reward = self.exploration_reward(game)

        isterminal = game.is_episode_finished()

        # Reset RNN state if game is over
        if isterminal:
            self.rnn_new_state = True
        else:
            self.rnn_new_state = False

        s2 = self.preprocess(game.get_state().screen_buffer) if not isterminal else None

        # Remember the transition that was just experienced.
        self.memory.add_transition(s1, a, s2, isterminal, reward)

        if learning_step % self.update_every == 0:
            self.learn_from_memory()

        return reward

    def get_position(self, game):
        return (game.get_game_variable(GameVariable.PLAYER_POSITION_X), game.get_game_variable(GameVariable.PLAYER_POSITION_Y))

    def distance(self, a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2) / 100

    def exploration_reward(self, game):

        pos = self.get_position(game)

        if len(self.positions) < 0:
            self.positions.append(pos)
            return 0

        weighted_sum = 0
        t = 0
        last_pos = ()
        for p in reversed(self.positions):
            if t == 0:
                last_pos = p
                weighted_sum += self.distance(pos, last_pos)
            else:
                new_distance = self.distance(pos, p)
                old_distance = self.distance(last_pos, p)
                diff = new_distance - old_distance
                weighted_diff = diff * (self.p_decay**t)
                weighted_sum += weighted_diff
            t += 1

        self.positions.append(pos)

        return weighted_sum


    def preprocess(self, img):
        """ Converts and down-samples the input image. """
        img = skimage.transform.resize(img, self.resolution)
        img = img.astype(np.float32)
        return img

    def learn(self, server, actions):

        saver = tf.train.Saver()
        if self.load_model:
            print("Loading model from: ", self.model_savefile)
            saver.restore(self.session, self.model_savefile)
        else:
            init = tf.initialize_all_variables()
            self.session.run(init)

        print("Starting the training!")

        time_start = time()
        train_results = []
        test_results = []

        game = server.start_game()
        for epoch in range(self.epochs):
            print("\nEpoch %d\n-------" % (epoch + 1))
            train_episodes_finished = 0
            train_scores = []
            game = server.restart_game(game)
            print("Training...")
            eps = self.exploration_rate(epoch, linear=True)
            print("Epsilon: " + str(eps))
            self.positions = []
            score = 0
            self.rnn_new_state = True
            for learning_step in trange(self.learning_steps_per_epoch):
                if game.is_player_dead():
                    if self.reward_exploration:
                        train_scores.append(score)
                        train_episodes_finished += 1
                        score = 0
                    self.positions = []
                    game.respawn_player()
                    self.rnn_new_state = True

                if game.is_episode_finished() or learning_step+1 == self.learning_steps_per_epoch:
                    if not self.reward_exploration:
                        score = game.get_total_reward()
                    train_scores.append(score)
                    game = server.restart_game(game)
                    train_episodes_finished += 1
                    self.positions = []
                    score = 0
                    self.rnn_new_state = True

                reward = self.perform_learning_step(game, actions, epoch, self.reward_exploration, learning_step)
                if self.reward_exploration:
                    score += reward

            print("%d training episodes played." % train_episodes_finished)

            train_scores = np.array(train_scores)

            print("Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()), \
                  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

            train_results.append((epoch, train_scores.mean(), train_scores.std()))

            print("\nTesting...")
            test_scores = []
            self.rnn_new_state = True
            for test_episode in trange(self.test_episodes_per_epoch):
                game = server.restart_game(game)
                self.positions = []
                score = 0
                while not game.is_episode_finished():
                    state = self.preprocess(game.get_state().screen_buffer)
                    if self.rnn_new_state:
                        best_action_index, s = self.fn_get_best_action(state)
                        self.rnn_state = s
                    else:
                        best_action_index, s = self.fn_get_best_action_rnn(state, self.rnn_state)
                        self.rnn_state = s
                    self.rnn_new_state = False
                    game.make_action(actions[best_action_index], self.frame_repeat)
                    if self.reward_exploration:
                        score += self.exploration_reward(game)
                if not self.reward_exploration:
                    score = game.get_total_reward()
                test_scores.append(score)
                self.rnn_new_state = True

            test_scores = np.array(test_scores)
            print("Results: mean: %.1f±%.1f," % (
                test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
                  "max: %.1f" % test_scores.max())

            test_results.append((epoch, test_scores.mean(), test_scores.std()))

            print("Saving the network weigths to:", self.model_savefile)
            saver.save(self.session, self.model_savefile)

            print("Saving the results...")
            with open("train_results.txt", "w") as train_result_file:
                train_result_file.write(str(train_results))
            with open("test_results.txt", "w") as test_result_file:
                test_result_file.write(str(test_results))

            print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

        print("======================================")
        print("Training finished.")

    def play(self, server, actions, episodes_to_watch=1):

        print("Loading model from: ", self.model_savefile)
        saver = tf.train.Saver()
        saver.restore(self.session, self.model_savefile)
        game = server.start_game()

        for _ in range(episodes_to_watch):
            game = server.restart_game(game)
            score = 0
            while not game.is_episode_finished():
                state = self.preprocess(game.get_state().screen_buffer)
                if self.rnn_new_state:
                    best_action_index, s = self.fn_get_best_action(state)
                else:
                    best_action_index, s = self.fn_get_best_action_rnn(state, self.rnn_state)
                self.rnn_new_state = False
                # Instead of make_action(a, frame_repeat) in order to make the animation smooth
                game.set_action(actions[best_action_index])
                for _ in range(self.frame_repeat):
                    game.advance_action()

                if self.reward_exploration:
                    score += self.position_reward(game, append=True)

            self.rnn_new_state = True

            # Sleep between episodes
            sleep(1.0)
            if not self.reward_exploration:
                score = game.get_total_reward()
            print("Total score: ", score)

        game.close()


class DoomServer:

    def __init__(self, screen_resolution, config_file_path, deathmatch=False, bots=7, visual=False, async=True):
        self.screen_resolution = screen_resolution
        self.deathmatch = deathmatch
        self.bots = bots
        self.visual = visual
        self.async = async
        self.config_file_path = config_file_path

    def start_game(self):
        print("Initializing doom...")
        game = DoomGame()
        game.load_config(self.config_file_path)
        game.set_window_visible(self.visual)
        game.set_screen_format(ScreenFormat.GRAY8)
        game.set_screen_resolution(self.screen_resolution)
        if self.deathmatch:
            #self.game.set_doom_map("map01")  # Limited deathmatch.
            game.set_doom_map("map02")  # Full deathmatch.
            # Start multiplayer game only with your AI (with options that will be used in the competition, details in cig_host example).
            game.add_game_args("-host 1 -deathmatch +timelimit 2.0 "
                               "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1")
            # Name your agent and select color
            # colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
            game.add_game_args("+name AI +colorset 0")

        if self.async:
            game.set_mode(Mode.ASYNC_PLAYER)
        else:
            game.set_mode(Mode.PLAYER)

        game.init()

        if self.deathmatch:
            game.send_game_command("removebots")
            for i in range(self.bots):
                game.send_game_command("addbot")

        #self.game.new_episode()

        print("Doom initialized.")
        return game

    def restart_game(self, game):
        if self.deathmatch:
            game.close()
            return self.start_game()
        game.new_episode()
        return game


# --------------- EXPERIMENTS ---------------

# Test settings
visual = False
async = False
screen_resolution = ScreenResolution.RES_320X240
scaled_resolution = (48, 64)

# Override these if used
p_decay = 1
bots = 7

# Super simple basic
'''
hidden_nodes = 1
conv1_filters = 1
conv2_filters = 1
replay_memory_size = 10
frame_repeat = 12
learning_steps_per_epoch = 100
test_episodes_per_epoch = 1
reward_exploration = False
epochs = 10
model_name = "super_simple_basic"
death_match = False
config = "../config/simpler_basic.cfg"
'''

# Simple basic
hidden_nodes = 128
conv1_filters = 8
conv2_filters = 8
replay_memory_size = 10000
frame_repeat = 12
learning_steps_per_epoch = 1000
test_episodes_per_epoch = 5
reward_exploration = False
epochs = 50
model_name = "simple_basic_dqrn"
death_match = False
config = "../config/simpler_basic.cfg"
update_every = 10

# Simple advanced
'''
hidden_nodes = 512
conv1_filters = 32
conv2_filters = 64
replay_memory_size = 1000000
frame_repeat = 4
learning_steps_per_epoch = 2000
test_episodes_per_epoch = 10
reward_exploration = False
epochs = 20
model_name = "simple_adv"
death_match = False
config = "../config/simpler_adv.cfg"
'''

# Simple exploration
'''
hidden_nodes = 512
conv1_filters = 32
conv2_filters = 64
replay_memory_size = 1000000
frame_repeat = 4
learning_steps_per_epoch = 2000
test_episodes_per_epoch = 10
reward_exploration = True
epochs = 50
model_name = "simple_exploration"
death_match = False
config = "../config/simpler_adv_expl.cfg"
p_decay = 0.90
'''

# Deathmatch exploration
'''
hidden_nodes = 512
conv1_filters = 32
conv2_filters = 64
replay_memory_size = 1000000
frame_repeat = 4
learning_steps_per_epoch = 5000
test_episodes_per_epoch = 10
reward_exploration = True
epochs = 200
model_name = "deathmatch_exploration_no_bots"
death_match = True
bots = 0
config = "../config/cig_train_expl.cfg"
p_decay = 0.90
'''

# ------------------------------------------------------------------
server = DoomServer(screen_resolution=screen_resolution,
                    config_file_path=config,
                    deathmatch=death_match,
                    visual=visual,
                    async=async,
                    bots=bots)

print("Starting game to get actions.")
game = server.start_game()
n = game.get_available_buttons_size()
actions = [list(a) for a in it.product([0, 1], repeat=n)]
game.close()
print("Game closed again")

script_dir = os.path.dirname(os.path.abspath(__file__)) #<-- absolute dir the script is in
print("Script path="+script_dir)

print("Creating learner")
learner = Learner(available_actions_count=len(actions),
                  frame_repeat=frame_repeat,
                  hidden_nodes=hidden_nodes,
                  conv1_filters=conv1_filters,
                  conv2_filters=conv2_filters,
                  epochs=epochs,
                  learning_steps_per_epoch=learning_steps_per_epoch,
                  test_episodes_per_epoch=test_episodes_per_epoch,
                  reward_exploration=reward_exploration,
                  resolution=scaled_resolution,
                  replay_memory_size=replay_memory_size,
                  p_decay=p_decay,
                  update_every=update_every,
                  model_savefile=script_dir+"/tf/"+model_name+".ckpt")

print("Training learner")
learner.learn(server, actions)

#learner.play(server, actions, episodes_to_watch=10)
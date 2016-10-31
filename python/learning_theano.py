#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from vizdoom import *
import itertools as it
import pickle
from random import sample, randint, random
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
from lasagne.init import HeUniform, Constant
from lasagne.layers import Conv2DLayer, InputLayer, DenseLayer, get_output, \
    get_all_params, get_all_param_values, set_all_param_values, GRULayer
from lasagne.nonlinearities import rectify
from lasagne.objectives import squared_error
from lasagne.updates import rmsprop
import theano
from theano import tensor
from tqdm import trange
import math
import experience_replay as er

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
        self.memory = er.ReplayMemory(capacity=replay_memory_size, resolution=resolution)

        # Create the input variables
        s1 = tensor.tensor4("State")
        a = tensor.vector("Action", dtype="int32")
        q2 = tensor.vector("Q2")
        r = tensor.vector("Reward")
        isterminal = tensor.vector("IsTerminal", dtype="int8")

        # Create the input layer of the network.
        l_in = InputLayer(shape=[None, 1, resolution[0], resolution[1]], input_var=s1)

        # Add 2 convolutional layers with ReLu activation
        conv1 = Conv2DLayer(l_in, num_filters=8, filter_size=[6, 6],
                          nonlinearity=rectify, W=HeUniform("relu"),
                          b=Constant(.1), stride=3)
        conv2 = Conv2DLayer(conv1, num_filters=8, filter_size=[3, 3],
                          nonlinearity=rectify, W=HeUniform("relu"),
                          b=Constant(.1), stride=2)

        # Add a single fully-connected layer.
        fc1 = DenseLayer(conv2, num_units=128, nonlinearity=rectify, W=HeUniform("relu"),
                         b=Constant(.1))

        rec = GRULayer(fc1, num_units=64)

        # Add the output layer (also fully-connected).
        # (no nonlinearity as it is for approximating an arbitrary real function)
        dqn = DenseLayer(rec, num_units=available_actions_count, nonlinearity=None)

        # Define the loss function
        q = get_output(dqn)
        # target differs from q only for the selected action. The following means:
        # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r
        target_q = tensor.set_subtensor(q[tensor.arange(q.shape[0]), a], r + discount_factor * (1 - isterminal) * q2)
        loss = squared_error(q, target_q).mean()

        # Update the parameters according to the computed gradient using RMSProp.
        params = get_all_params(dqn, trainable=True)
        updates = rmsprop(loss, params, learning_rate)

        # Compile the theano functions
        print("Compiling the network ...")
        function_learn = theano.function([s1, q2, a, r, isterminal], loss, updates=updates, name="learn_fn")
        function_get_q_values = theano.function([s1], q, name="eval_fn")
        function_get_best_action = theano.function([s1], tensor.argmax(q), name="test_fn")
        print("Network compiled.")

        def simple_get_best_action(state):
            return function_get_best_action(state.reshape([1, 1, resolution[0], resolution[1]]))

        self.net = dqn
        self.fn_learn = function_learn
        self.fn_get_q_values = function_get_q_values
        self.fn_get_best_action = simple_get_best_action


    def learn_from_memory(self):
        """ Learns from a single transition (making use of replay memory).
        s2 is ignored if s2_isterminal """

        # Get a random minibatch from the replay memory and learns from it.
        if self.memory.size > self.batch_size:
            s1, a, s2, isterminal, r = self.memory.get_sample(self.batch_size)

            q2 = np.max(self.fn_get_q_values(s2), axis=1)
            # the value of q2 is ignored in learn if s2 is terminal
            self.fn_learn(s1, q2, a, r, isterminal)


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

        game.set_window_visible(visual)
        game.set_mode(Mode.PLAYER)
        game.init()

        if self.load_model:
            # Load the network's parameters from a file
            params = pickle.load(open(self.model_savefile, "rb"))
            set_all_param_values(self.net, params)

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
                test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(), "max: %.1f" % test_scores.max())

            print("Saving the network weigths to:", self.model_savefile)
            pickle.dump(get_all_param_values(self.net), open(self.model_savefile, "wb"))

            print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

        game.close()
        print("======================================")
        print("Loading the network weigths from:", self.model_savefile)
        print("Training finished.")

    def play(self, game, actions, episodes_to_watch=1, reward_exploration=False):

        game.set_window_visible(True)
        game.set_mode(Mode.ASYNC_PLAYER)
        game.init()

        # Load the network's parameters from a file
        params = pickle.load(open(self.model_savefile, "rb"))
        set_all_param_values(self.net, params)

        # Reinitialize the game with window visible
        game.set_window_visible(True)
        game.set_mode(Mode.ASYNC_PLAYER)
        game.init()

        for _ in range(episodes_to_watch):
            game.new_episode()
            score = 0
            while not game.is_episode_finished():
                state = self.preprocess(game.get_state().screen_buffer)
                best_action_index = self.fn_get_best_action(state)
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
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from __future__ import division
from vizdoom import *
import itertools as it
import pickle
from random import sample, randint, random
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
from lasagne.init import HeUniform, Constant
from lasagne.layers import Conv2DLayer, InputLayer, DenseLayer, LSTMLayer, get_output, \
    get_all_params, get_all_param_values, set_all_param_values
from lasagne.nonlinearities import rectify
from lasagne.objectives import squared_error
from lasagne.updates import rmsprop
import theano
from theano import tensor
from tqdm import trange
from vizdoom import ScreenResolution as res
from random import choice


learning_rate = 0.00025
# learning_rate = 0.0001
discount_factor = 0.99
epochs = 100
#learning_steps_per_epoch = 2000
learning_steps_per_epoch = 20000
replay_memory_size = 10000

# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 1000
#test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 4
resolution = (30, 45)
episodes_to_watch = 10

# Visual control
visual_training = False
visualize_result = False

# Scores and rewards gained during an episode
scores = []
rewards = []

# Results of training
train_results = []
test_results = []

# Converts and downsamples the input image
first_print = []


def preprocess(img):
    img = img[0]
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img

class ReplayMemory:
    def __init__(self, capacity):
        state_shape = (capacity, 1, resolution[0], resolution[1])
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.bool_)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos, 0] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, 0] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]


def create_network(available_actions_count):
    # Create the input variables
    s1 = tensor.tensor4("States")
    a = tensor.vector("Actions", dtype="int32")
    q2 = tensor.vector("Next State's best Q-Value")
    r = tensor.vector("Rewards")
    isterminal = tensor.vector("IsTerminal", dtype="int8")

    # Create the input layer of the network.
    l_in = InputLayer(shape=[None, 1, resolution[0], resolution[1]], input_var=s1)

    # Add 2 convolutional layers with ReLu activation
    l_conv1 = Conv2DLayer(l_in, num_filters=32, filter_size=[8, 8],
                      nonlinearity=rectify, W=HeUniform("relu"),
                      b=Constant(.1), stride=4)
    l_conv2 = Conv2DLayer(l_conv1, num_filters=64, filter_size=[4, 4],
                      nonlinearity=rectify, W=HeUniform("relu"),
                      b=Constant(.1), stride=2)

    # Add a fully-connected layer.
    l_hid1 = DenseLayer(l_conv2, num_units=4608, nonlinearity=rectify, W=HeUniform("relu"),
                     b=Constant(.1))

    # LSTM layer
    # l_lstm = LSTMLayer(l_hid1, num_units=512, grad_clipping=10)

    # Add the output layer (also fully-connected).
    # (no nonlinearity as it is for approximating an arbitrary real function)
    dqn = DenseLayer(l_hid1, num_units=available_actions_count, nonlinearity=None)

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

    # Returns Theano objects for the net and functions.
    return dqn, function_learn, function_get_q_values, simple_get_best_action


def learn_from_transition(s1, a, s2, s2_isterminal, r):
    """ Learns from a single transition (making use of replay memory).
    s2 is ignored if s2_isterminal """

    # Remember the transition that was just experienced.
    memory.add_transition(s1, a, s2, s2_isterminal, r)

    # Get a random minibatch from the replay memory and learns from it.
    if memory.size > batch_size:
        s1, a, s2, isterminal, r = memory.get_sample(batch_size)
        q2 = np.max(get_q_values(s2), axis=1)
        # the value of q2 is ignored in learn if s2 is terminal
        learn(s1, q2, a, r, isterminal)


def get_score():
    health = max(0,game.get_game_variable(GameVariable.HEALTH))
    frags = game.get_game_variable(GameVariable.FRAGCOUNT)
    return frags + health


def get_reward():
    new_score = get_score()
    old_score = 0
    if len(scores) is 0:
        old_score = new_score
    else:
        old_score = scores[-1]
    diff = new_score - old_score
    scores.append(new_score)
    rewards.append(diff)
    #if diff is not 0:
        #print("REWARDED WITH " + str(diff))
    return diff


def perform_learning_step(epoch):
    """ Makes an action according to eps-greedy policy, observes the result
    (next state, reward) and learns from the transition"""

    def exploration_rate(epoch):
        """# Define exploration rate change over time"""
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.1 * epochs  # 10% of learning time
        eps_decay_epochs = 0.6 * epochs  # 60% of learning time

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return start_eps - (epoch - const_eps_epochs) / \
                               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps

    s1 = preprocess(game.get_state().image_buffer)

    # With probability eps make a random action.
    eps = exploration_rate(epoch)
    if random() <= eps:
        a = randint(0, len(actions) - 1)
    else:
        # Choose the best action according to the network.
        a = get_best_action(s1)
    #reward = game.make_action(actions[a], frame_repeat)
    game.make_action(actions[a], frame_repeat)
    r = get_reward()
    isterminal = game.is_episode_finished()
    s2 = preprocess(game.get_state().image_buffer) if not isterminal else None

    learn_from_transition(s1, a, s2, isterminal, r)


# ---------- Game Server ---------

game = DoomGame()

game.set_vizdoom_path("../../bin/vizdoom")

# Use CIG example config or your own.
game.load_config("../config/cig_train.cfg")
game.set_screen_resolution(res.RES_320X240)
game.set_window_visible(visual_training)

# Select game and map you want to use.
game.set_doom_game_path("../../scenarios/freedoom2.wad")
# game.set_doom_game_path("../../scenarios/doom2.wad")  # Not provided with environment due to licences

game.set_doom_map("map01")  # Limited deathmatch.
# game.set_doom_map("map02")  # Full deathmatch.

# Start multiplayer game only with your AI (with options that will be used in the competition, details in cig_host example).
game.add_game_args("-host 1 -deathmatch +timelimit 2.0 "
                   "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1")

# Name your agent and select color
# colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
game.add_game_args("+name AI +colorset 0")

# Multiplayer requires the use of asynchronous modes, but when playing only with bots, synchronous modes can also be used.
# game.set_mode(Mode.PLAYER)

# game.set_window_visible(False)

game.init()

# Three example sample actions
actions = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0],    # TURN_LEFT
    [0, 1, 0, 0, 0, 0, 0, 0, 0],    # TURN_RIGHT
    [0, 0, 1, 0, 0, 0, 0, 0, 0],    # ATTACK
    [0, 0, 0, 1, 0, 0, 0, 0, 0],    # MOVE_RIGHT
    [0, 0, 0, 0, 1, 0, 0, 0, 0],    # MOVE_LEFT
    [0, 0, 0, 0, 0, 1, 0, 0, 0],    # MOVE_FORWARD
    [0, 0, 0, 0, 0, 0, 1, 0, 0]     # MOVE_BACKWARD
]

# Play with this many bots
bots = 7

# --------------------------------

# Create replay memory which will store the transitions
memory = ReplayMemory(capacity=replay_memory_size)

net, learn, get_q_values, get_best_action = create_network(len(actions))

def new_game():
    # Add specific number of bots
    # (file examples/bots.cfg must be placed in the same directory as the Doom executable file,
    # edit this file to adjust bots).
    #print("Starting a new game.\n")
    game.send_game_command("removebots")
    for i in range(bots):
        game.send_game_command("addbot")
    game.new_episode()
    del scores[:] # Empty score list
    del rewards[:] # Empty reward list
    #print("New game started with " + str(bots) + " bots\n")


print("Starting the training!")

time_start = time()
for epoch in range(epochs):
    print("\nEpoch %d\n-------" % (epoch + 1))
    train_episodes_finished = 0
    train_scores = []

    print("Training...")

    new_game()
    for learning_step in trange(learning_steps_per_epoch):

        if game.is_player_dead():
            score = sum(rewards)
            #print("Score of episode: " + str(score))
            train_scores.append(score)
            train_episodes_finished += 1
            # Use this to respawn immediately after death, new state will be available.
            game.respawn_player()
            del scores[:] # Empty score list
            del rewards[:] # Empty reward list

        if game.is_episode_finished():
            score = sum(rewards)
            #print("Score of episode: " + str(score))
            train_scores.append(score)
            train_episodes_finished += 1
            new_game()

        perform_learning_step(epoch)

    # Get score from current episode
    score = sum(rewards)
    #print("Score of episode: " + str(score))
    train_scores.append(score)
    train_episodes_finished += 1

    print("%d training episodes played." % train_episodes_finished)

    train_scores = np.array(train_scores)

    print("Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()), \
        "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

    # Add to result list
    train_results.append(train_scores.mean())

    print("Saving the results...")
    with open("train_results.txt", "w") as train_result_file:
        train_result_file.write(str(train_results))

    print("\nTesting...")
    test_episode = []
    test_scores = []
    new_game()
    for test_episode in trange(test_episodes_per_epoch):

        reward = get_reward()
        if reward is not 0:
            rewards.append(reward)
            #print("REWARDED WITH " + str(reward))

        if game.is_player_dead():
            score = sum(rewards)
            #print("Score of episode: " + str(score))
            test_scores.append(score)
            # Use this to respawn immediately after death, new state will be available.
            game.respawn_player()
            del scores[:] # Empty score list
            del rewards[:] # Empty reward list

        if game.is_episode_finished():
            score = sum(rewards)
            #print("Score of episode: " + str(score))
            test_scores.append(score)
            new_game()

        state = preprocess(game.get_state().image_buffer)
        best_action_index = get_best_action(state)
        game.make_action(actions[best_action_index], frame_repeat)

    # Get score from current episode
    score = sum(rewards)
    #print("Score of episode: " + str(score))
    test_scores.append(score)

    test_scores = np.array(test_scores)
    print("Results: mean: %.1f±%.1f," % (
        test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(), "max: %.1f" % test_scores.max())

    # Add to result list
    test_results.append(test_scores.mean())

    print("Saving the network weigths...")
    pickle.dump(get_all_param_values(net), open('weights.dump', "w"))

    print("Saving the results...")
    with open("test_results.txt", "w") as test_result_file:
        test_result_file.write(str(test_results))

    print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

game.close()
print("======================================")
print("Training finished. It's time to watch!")

# Load the network's parameters from a file
params = pickle.load(open('weights.dump', "r"))
set_all_param_values(net, params)

# Reinitialize the game with window visible
if visualize_result:
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()

    for _ in range(episodes_to_watch):
        new_game()
        while not game.is_episode_finished():

            reward = get_reward()
            if reward is not 0:
                rewards.append(reward)
                #print("REWARDED WITH " + str(reward))

            state = preprocess(game.get_state().image_buffer)
            best_action_index = get_best_action(state)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = sum(rewards)
        #print("Score of episode: " + str(score))
        print("Total score: ", str(score))

# ----------------------------

game.close()
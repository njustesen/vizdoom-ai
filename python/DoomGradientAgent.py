#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from __future__ import division
import itertools as it
import pickle
from random import sample, randint, random
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
from lasagne.init import HeUniform, Constant
from lasagne.layers import Conv2DLayer, InputLayer, DenseLayer, LSTMLayer, get_output, \
    get_all_params, get_all_param_values, set_all_param_values
from lasagne.nonlinearities import rectify,  tanh, softmax
from lasagne.objectives import squared_error
from lasagne.updates import rmsprop
import lasagne
import theano
from theano import tensor
from tqdm import trange
from random import choice
import DoomBotServer

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


class DoomGradientAgent(object):
    """
    Reinforcement Learning Agent

    This agent can learn to solve reinforcement learning tasks from
    OpenAI Gym by applying the policy gradient method.
    """

    def __init__(self, n_outputs):

        self.n_outputs = n_outputs

        # symbolic variables for state, action, and advantage
        #sym_state = tensor.fmatrix()
        sym_state = tensor.tensor4("States")
        sym_action = tensor.ivector()
        sym_advantage = tensor.fvector()

        # Create the input layer of the network.
        l_in = InputLayer(shape=[None, 1, resolution[0], resolution[1]], input_var=sym_state)
        l_conv1 = Conv2DLayer(l_in, num_filters=32, filter_size=[8, 8],
                          nonlinearity=rectify, W=HeUniform("relu"),
                          b=Constant(.1), stride=4, name='convLayer1')
        l_conv2 = Conv2DLayer(l_conv1, num_filters=64, filter_size=[4, 4],
                          nonlinearity=rectify, W=HeUniform("relu"),
                          b=Constant(.1), stride=2, name='convLayer2')
        l_hid1 = DenseLayer(l_conv2, num_units=4608, nonlinearity=rectify, W=HeUniform("relu"),
                     b=Constant(.1), name='hiddenLayer1')
        # LSTM layer
        # l_lstm = LSTMLayer(l_hid1, num_units=512, grad_clipping=10)
        l_out = DenseLayer(l_hid1, num_units=n_outputs, nonlinearity=softmax, name='outputlayer')

        # Define the loss function
        eval_out = get_output(l_out, {l_in: sym_state}, deterministic=True)

        # get trainable parameters in the network.
        params = lasagne.layers.get_all_params(l_out, trainable=True)
        # get total number of timesteps
        t_total = sym_state.shape[0]
        # loss function that we'll differentiate to get the policy gradient
        loss = -tensor.log(eval_out[tensor.arange(t_total), sym_action]).dot(sym_advantage) / t_total
        # learning_rate
        learning_rate = tensor.fscalar()
        # get gradients
        grads = tensor.grad(loss, params)
        # update function
        updates = lasagne.updates.sgd(grads, params, learning_rate=learning_rate)
        # declare training and evaluation functions
        self.f_train = theano.function([sym_state, sym_action, sym_advantage, learning_rate], loss, updates=updates, allow_input_downcast=True)
        self.f_eval = theano.function([sym_state], eval_out, allow_input_downcast=True)

    def learn(self, n_epochs=100, n_deaths=10,
              learning_rate=0.1, discount_factor=1.0, n_early_stop=0):
        """
        Learn the given environment by the policy gradient method.
        """
        self.mean_train_rs = []
        self.mean_val_rs = []
        self.loss = []

        # Make Doom Game
        server = DoomBotServer(graphics=False, fast=True)

        for epoch in xrange(n_epochs):
            # 1. collect trajectories until we have at least t_per_batch total timesteps
            trajs = []; t_total = 0
            server.reset()
            server.game.new_episode()
            for i in n_deaths:
                traj = self.get_trajectory(server.game, actions=server.actions, deterministic=False)
                trajs.append(traj)
                t_total += len(traj["r"])
            all_s = np.concatenate([traj["s"] for traj in trajs])

            # 2. compute cumulative discounted rewards (returns)
            rets = [self._cumulative_discount(traj["r"], discount_factor) for traj in trajs]
            maxlen = max(len(ret) for ret in rets)
            padded_rets = [np.concatenate([ret, np.zeros(maxlen-len(ret))]) for ret in rets]
            # 3. compute time-dependent baseline
            baseline = np.mean(padded_rets, axis=0)
            # 4. compute advantages
            advs = [ret - baseline[:len(ret)] for ret in rets]
            all_a = np.concatenate([traj["a"] for traj in trajs])
            all_adv = np.concatenate(advs)

            # 5. do policy gradient update step
            loss = self.f_train(all_s, all_a, all_adv, learning_rate)
            train_rs = np.array([traj["r"].sum() for traj in trajs]) # trajectory total rewards
            eplens = np.array([len(traj["r"]) for traj in trajs]) # trajectory lengths
            # compute validation reward
            val_rs = np.array([self.get_trajectory(env, deterministic=True)['r'].sum() for _ in range(10)])
            # update stats
            self.mean_train_rs.append(train_rs.mean())
            self.mean_val_rs.append(val_rs.mean())
            self.loss.append(loss)
            # print stats
            print('%3d mean_train_r: %6.2f mean_val_r: %6.2f loss: %f' % (epoch+1, train_rs.mean(), val_rs.mean(), loss))
            # render solution
            #self.get_trajectory(env, traj_t_limit, render=True)
            # check for early stopping: true if the validation reward has not changed in n_early_stop epochs
            if n_early_stop and len(self.mean_val_rs) >= n_early_stop and \
                all([x == self.mean_val_rs[-1] for x in self.mean_val_rs[-n_early_stop:-1]]):
                break

    def get_trajectory(self, game, actions, render=False, deterministic=True):
        """
        Compute trajectroy by iteratively evaluating the agent policy on the environment.
        """
        traj = {'s': [], 'a': [], 'r': [],}
        while True:
            if game.is_player_dead():
                break
            if game.is_episode_finished():
                game.new_episode()
            state = self.preprocess(game.get_state().image_buffer)
            action = self.get_action(state, deterministic)
            reward = game.make_action(action, frame_repeat)
            traj['s'].append(state)
            traj['a'].append(action)
            traj['r'].append(reward)
        return {'s': np.array(traj['s']), 'a': np.array(traj['a']), 'r': np.array(traj['r'])}

    def get_action(self, state, deterministic=True):
        """
        Evaluate the agent policy to choose an action, a, given state, s.
        """
        # compute action probabilities
        prob_a = self.f_eval(state.reshape(1,-1))
        if deterministic:
            # choose action with highest probability
            return prob_a.argmax()
        else:
            # sample action from distribution
            return (np.cumsum(np.asarray(prob_a)) > np.random.rand()).argmax()

    def _cumulative_discount(self, r, gamma):
        """
        Compute the cumulative discounted rewards (returns).
        """
        r_out = np.zeros(len(r), 'float64')
        r_out[-1] = r[-1]
        for i in reversed(xrange(len(r)-1)):
            r_out[i] = r[i] + gamma * r_out[i+1]
        return r_out

    def preprocess(img):
        img = img[0]
        img = skimage.transform.resize(img, resolution)
        img = img.astype(np.float32)
        return img


# --------------------------------



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
            # Use this to respawn immediately after death, new state will be available.
            game.respawn_player()

        if game.is_episode_finished():
            score = sum(rewards)
            #print("Score of episode: " + str(score))
            train_scores.append(score)
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
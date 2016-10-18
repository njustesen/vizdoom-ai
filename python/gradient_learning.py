#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from vizdoom import *
import itertools as it
import pickle
from random import sample, randint, random
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
from lasagne.init import HeUniform, Constant
from lasagne.layers import Conv2DLayer, InputLayer, DenseLayer, get_output, \
    get_all_params, get_all_param_values, set_all_param_values
from lasagne.nonlinearities import rectify, softmax
from lasagne.objectives import squared_error
from lasagne.updates import rmsprop
import theano
from theano import tensor
from tqdm import trange
import lasagne

# Configuration file path
config_file_path = "../../examples/config/simpler_basic.cfg"
graphics = True
actions = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0],    # TURN_LEFT
    [0, 1, 0, 0, 0, 0, 0, 0, 0],    # TURN_RIGHT
    [0, 0, 1, 0, 0, 0, 0, 0, 0]     # ATTACK
]

class DoomGradientAgent(object):
    """
    Reinforcement Learning Agent

    This agent can learn to solve reinforcement learning tasks from
    OpenAI Gym by applying the policy gradient method.
    """

    def __init__(self, n_outputs, resolution, frame_repeat):

        self.n_outputs = n_outputs
        self.frame_repeat = frame_repeat
        self.resolution = resolution

        # Three example sample actions

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
        l_hid1 = DenseLayer(l_conv2, num_units=128, nonlinearity=rectify, W=HeUniform("relu"),
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

    def learn(self, n_epochs=100, n_runs=100,
              learning_rate=0.1, discount_factor=1.0, n_early_stop=0):
        """
        Learn the given environment by the policy gradient method.
        """
        self.mean_train_rs = []
        self.mean_val_rs = []
        self.loss = []

        # Make Doom Game
        # Create Doom instance
        game = DoomGame()
        game.load_config(config_file_path)
        game.set_window_visible(graphics)
        game.set_mode(Mode.PLAYER)
        game.init()

        for epoch in xrange(n_epochs):
            # 1. collect trajectories until we have at least t_per_batch total timesteps
            trajs = []; t_total = 0
            for i in range(n_runs):
                traj = self.get_trajectory(game, deterministic=False)
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
            val_rs = np.array([self.get_trajectory(game, deterministic=True)['r'].sum() for _ in range(10)])
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

    def get_trajectory(self, game, render=False, deterministic=True):
        """
        Compute trajectroy by iteratively evaluating the agent policy on the environment.
        """
        traj = {'s': [], 'a': [], 'r': [],}
        game.new_episode()
        while not game.is_episode_finished():
            state = self.preprocess(game.get_state().image_buffer)
            action = self.get_action(state, deterministic)
            reward = game.make_action(actions[action], self.frame_repeat)
            traj['s'].append(state)
            traj['a'].append(action)
            traj['r'].append(reward)
        return {'s': np.array(traj['s']), 'a': np.array(traj['a']), 'r': np.array(traj['r'])}

    def get_action(self, state, deterministic=True):
        """
        Evaluate the agent policy to choose an action, a, given state, s.
        """
        # compute action probabilities
        prob_a = self.f_eval(state.reshape([1, 1, self.resolution[0], self.resolution[1]]))
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

    def preprocess(self, img):
        img = img[0]
        # Reshape from (x,y) to (channels, x, y)
        img = skimage.transform.resize(img, self.resolution)
        img = img.astype(np.float32)
        img = img[np.newaxis, ...]
        return img


agent = DoomGradientAgent(n_outputs = 3, resolution = (30, 45), frame_repeat = 12)
agent.learn()
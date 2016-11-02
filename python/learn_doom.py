#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from vizdoom import *
import itertools as it
import tf_learner as tfl
import os
import doom_server as ds

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in

# CONFIGURATIONS

# Test settings
visual = False
async = False
screen_resolution = ScreenResolution.RES_320X240
scaled_resolution = (48, 64)

# Simple basic
'''
hidden_nodes = 128
conv1_filters = 8
conv2_filters = 8
replay_memory_size = 10000
frame_repeat = 12
learning_steps_per_epoch = 1000
test_episodes_per_epoch = 10
reward_exploration = False
epochs = 20
model_name = "simple_basic"
death_match = False
config = "../config/simpler_basic.cfg"
'''

# Simple advanced
hidden_nodes = 4608
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

# Deathmatch exploration
'''
hidden_nodes = 4608
conv1_filters = 32
conv2_filters = 64
replay_memory_size = 10000
frame_repeat = 4
learning_steps_per_epoch = 5000
test_episodes_per_epoch = 10
reward_exploration = True
epochs = 100
model_name = "deathmatch_exploration"
death_match = True
config = "../config/cig_train.cfg"
'''

# ------------------------------------------------------------------
server = ds.DoomServer(screen_resolution=screen_resolution,
                       config_file_path=config,
                       deathmatch=death_match,
                       visual=visual,
                       async=async)

print("Starting game to get actions.")
game = server.start_game()
n = game.get_available_buttons_size()
actions = [list(a) for a in it.product([0, 1], repeat=n)]
game.close()
print("Game closed again")

print("Creating learner")
learner = tfl.Learner(available_actions_count=len(actions),
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
                      model_savefile=script_dir+"/tf/"+model_name+".ckpt")

print("Training learner")
learner.learn(server, actions)

#learner.play(server, actions, episodes_to_watch=10)

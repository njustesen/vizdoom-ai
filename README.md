# vizdoom-ai

This code in this project trains a deep convolutional neural network using Deep Q Networks or Gradient Policy to learn a policy in the ViZDoom environment. It's main contributions are the two-phase learning method of first learning to navigate and then learning to shoot towards enemy bots. The core part of this project can be found in learn_doom_resume.py.

To run this code you must first setup ViZDoom and then clone this repository into the ViZDoom folder. Then copy the compiled vizdoom.so file into the cloned folder.

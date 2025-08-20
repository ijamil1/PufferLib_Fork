forked PufferLib

Created my own multi-agent env and trained it in a vectorized/parallel fashion using PufferLib's PPO implementation


IMPORTANT NOTE: the majority of the code in this branch is just a result of forking the actual PufferLib repo. These are the exact files I created:

pufferlib/config/ocean/sharks_and_minnows.ini: configuration file that PufferLib's PPO trainer uses to initialize hyperparameters

pufferlib/ocean/sharks_and_minnows/sharks_and_minnows.py: defines my custom environment's class. This class is instantiated prior to training

pufferlib/ocean/sharks_and_minnows/sharks_and_minnows.h: contains the functionality of the custom environment such as defining functions to handle sharks' movement, minnows' movement, state updates, reward assignment, episode reset/initialization, environment rendering, and so on.

pufferlib/ocean/sharks_and_minnows/sharks_and_minnows.c: used this to test/debug .h file prior to Python binding

pufferlib/ocean/sharks_and_minnows/binding.c: necessary file to succesfully bind the C and Python files

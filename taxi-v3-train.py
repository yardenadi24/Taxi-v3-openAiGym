#imports:
import gym
import numpy as np
import random

#env initialize:
env = gym.make("Taxi-v3")

#how many action for each state
actionSize = env.action_space.n

#how many states there is
stateSize = env.observation_space.np_random

#discount factor:
discount = 0.95
import numpy as np
import os
import gym

env = gym.make("Pong-v0")
env = gym.wrappers.Monitor(env, "recording", force=True)
env.reset()
for _ in range(5000):
    env.step(env.action_space.sample())
    env.render('human')
env.close()
env.env.close()


import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import Model

import tensorflow as tf

import gymnasium as gym

class DQN(Model):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()

        self.replay_memory = []

        self.dense1 = Dense(48, activation="tanh", input_dim=state_size)
        self.dense2 = Dense(action_size, activation="softmax")

    def call(self, x):
        x = self.dense1(x)
        return self.dense2(x)
    
    
    def remember(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))


env = gym.make('MountainCar-v0', render_mode="human")

model = DQN(env.observation_space.shape[0], env.action_space.n)
model.load_weights("mountaincar2")  # 경로 항상 유의 (경로 기준은 터미널)

state, info = env.reset()

while True:
    state, info = env.reset()

    for step in range(1000):
        action_list = model.call(np.array([state])).numpy()[0]
        action = np.argmax(action_list)

        state, _, done, _, _ = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(step+1))
            break

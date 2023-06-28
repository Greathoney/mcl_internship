import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import Model

import tensorflow as tf

import gymnasium as gym

class DQN(Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.d1 = Dense(64, input_dim=4, activation='tanh')
        self.d2 = Dense(2, activation='linear')
        self.optimizer = Adam(0.001)

        self.M = []  # M은 리플레이 버퍼

    def call(self, x): # x는 넘파이 어레이
        x = self.d1(x)
        y_hat = self.d2(x)
        return y_hat  # y_hat은 텐서 (-1x2)

model = DQN()
model.load_weights("cartpole/model")  # 경로 항상 유의 (경로 기준은 터미널)

env = gym.make('CartPole-v1', render_mode="human")
state, info = env.reset()

for step in range(1000):
    action_list = model.call(np.array([state])).numpy()[0]
    action = np.argmax(action_list)

    state, _, done, _, _ = env.step(action)

    if done:
        print("Episode finished after {} timesteps".format(step+1))
        break


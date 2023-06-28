import random

import gymnasium as gym
import numpy as np

import tensorflow as tf

from keras import Model
from keras.layers import Dense

class DQN(Model):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()

        self.replay_memory = []

        self.dense1 = Dense(100, activation="relu", input_dim=state_size)
        self.dense2 = Dense(100, activation="relu")
        self.dense3 = Dense(100, activation="relu")
        self.dense4 = Dense(100, activation="relu")
        self.dense5 = Dense(action_size, activation="softmax")

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        # self.optimizer_side = tf.keras.optimizers.Adam(learning_rate=0.0003)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)

        return x
    
    def remember(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))


model = DQN(3, 9)
model.load_weights("model2")

env = gym.make("Pendulum-v1", render_mode="human")


for episode in range(10):
    state, info = env.reset()
    terminated = False
    truncated = False
    step = 0

    rewards = []
    before_reward = None

    while not terminated and step < 1000:

        # 모델로 행동 예측
        action = model.call(np.array([state])).numpy()[0]  # type: ignore
        # if np.random.rand() < 0.01:
        #     action = np.random.choice(9)
        # else:
        #     action = np.argmax(action)
        # action = np.random.choice(9, p=action)
        action = np.argmax(action)
        # 행동 실행
        next_state, reward, terminated, truncated, info = env.step((action/2-2,))

        state = next_state
        step += 1

        rewards.append(reward)
    
    print("Episode: {}, Steps: {}, Score: {:.2f}, Last Score: {:.2f}".format(episode, step, sum(rewards) / len(rewards), sum(rewards[-10:])/10))

env.close()
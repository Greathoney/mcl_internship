import random
import gymnasium as gym # pip install gymnasium[classic-control]

import numpy as np
import tensorflow as tf 

from keras import Model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import mean_squared_error

class DQN(Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.d1 = Dense(64, input_dim=4, activation='relu')
        self.d2 = Dense(32, activation='relu')
        self.d3 = Dense(32, activation='relu')
        self.d3 = Dense(2, activation='linear')
        self.optimizer = Adam(0.001)

        self.memory = []

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        y_hat = self.d3(x)
        return y_hat
    

model_left = DQN()
model_right = DQN()

model_left.load_weights('cartpole2/model_left2')
model_right.load_weights('cartpole2/model_right2')

model = {
    'left': model_left,
    'right': model_right
}

# 카트폴 게임 환경 생성
# env = gym.make('CartPole-v1')
env = gym.make('CartPole-v1', render_mode="human")

# 인퍼런스
for episode in range(10):
    state, info = env.reset()
    dir = np.random.choice(['left'])
    print("Episode {} started".format(episode))
    print("Init dir: {}".format(dir))

    for step in range(3000):
        action_list = model[dir].call(np.array([state])).numpy()[0]  # type: ignore
        # 확률적으로 행동을 선택
        action = np.argmax(action_list)

        next_state, _, _, _, _ = env.step(action)
        done = not -0.3 < next_state[2] < 0.3

        if done:
            print("Episode {} done after {} steps \n".format(episode, step))
            break

        stable_condition = np.all(abs(np.array(next_state[1:])) < 0.2)

        if dir == 'left' and next_state[0] < -0.8 and stable_condition:
            dir = 'right'
            print("dir changed to right")


        if dir == 'right' and next_state[0] > 0.8 and stable_condition:
            dir = 'left'
            print("dir changed to left")

        state = next_state

    
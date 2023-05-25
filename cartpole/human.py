"""
CartPole-v1 게임을 플레이하는 코드입니다.
키보드의 k와 l을 눌러서 왼쪽과 오른쪽으로 이동할 수 있습니다.
"""


import gymnasium as gym
import time

action = 0

##############################################
# 키보드 입력을 받아 action을 결정하는 코드
from pynput import keyboard  # pip install pynput

def left():
    global action
    action = 0

def right():
    global action
    action = 1


listener = keyboard.GlobalHotKeys({
    'k': left,  # k는 왼쪽으로 이동
    'l': right  # l은 오른쪽으로 이동
})

listener.start()
##############################################

env = gym.make('CartPole-v1', render_mode="human")
env.reset()
print("READY!")
time.sleep(2)

score = 0

while True:
    # env.step 진행
    _, _, done, _, _ = env.step(action)

    if done:
        print("GAME OVER! score: {}".format(score))
        time.sleep(1)
        break

    score += 1
    time.sleep(0.1)
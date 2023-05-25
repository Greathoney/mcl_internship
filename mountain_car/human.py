##############################################
# 키보드 입력을 받아 action을 결정하는 코드
from pynput import keyboard  # pip install pynput

action = 0

def left():
    global action
    action = 0

def right():
    global action
    action = 2

def dont_accelerate():
    global action
    action = 1


listener = keyboard.GlobalHotKeys({
    'k': left,  # k는 왼쪽으로 이동
    'l': right,  # l은 오른쪽으로 이동
    'o': dont_accelerate  # o는 가속하지 않음
})

listener.start()
##############################################

import gymnasium as gym
import time

env = gym.make('MountainCar-v0', render_mode="human")
env.reset()
print("READY!")
time.sleep(2)

steps = 0

while True:
    # env.step 진행
    _, _, done, _, _ = env.step(action)

    if done:
        print("steps: {}".format(steps))
        time.sleep(1)
        break

    steps += 1
    time.sleep(0.1)
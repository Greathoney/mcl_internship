##############################################
# 키보드 입력을 받아 action을 결정하는 코드
from pynput import keyboard  # pip install pynput

action = 0

def left():
    global action
    action = -2

def right():
    global action
    action = 2

def dont_accelerate():
    global action
    action = 0


listener = keyboard.GlobalHotKeys({
    'j': left,  # j는 시계 방향으로 가속
    'l': right,  # l은 반시계 방향으로 가속 
    'k': dont_accelerate  # k는 가속하지 않음
})

listener.start()
##############################################

import gymnasium as gym
import time

env = gym.make('Pendulum-v1', render_mode="human")
state, info = env.reset()
steps = 0

while True:
    # env.step 진행
    next_state, reward, done, _, _ = env.step((action,))

    print("현재", steps, "스텝 에서의 보상 반환값:", reward)

    next_state = state

    steps += 1
    time.sleep(0.1)
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
cart_goal = 'left'
print("GO! 카트를 왼쪽으로 옮기세요!")

while True:
    # env.step 진행
    next_state, _, done, _, _ = env.step(action)
    # print("현재 위치 {:.2f}, 카트 속도 {:.2f}, 각도 {:.2f}, 각속도 {:.2f}".format(*next_state))
    
    if cart_goal == 'left' and next_state[0] < -1:
        score += 1
        cart_goal = 'right'
        print("득점! 이제 카트를 오른쪽으로 옮기세요!")

    if cart_goal == 'right' and next_state[0] > 1:
        score += 1
        cart_goal = 'left'
        print("득점! 이제 카트를 왼쪽으로 옮기세요!")

    if not -0.418 < next_state[2] < 0.418:  # 좌우 각도 24도 이상
        # done에서의 각도 제한 보다 더 큰 각도 허용하기
        print("GAME OVER! score: {}".format(score))
        time.sleep(1)
        break

    time.sleep(0.1)
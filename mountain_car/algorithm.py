import gymnasium as gym

# env = gym.make("MountainCar-v0")
env = gym.make("MountainCar-v0", render_mode="human")

for episode in range(1000):
    state, info = env.reset()
    terminated = False
    truncated = False
    step = 0
    
    max_score = state[0]
    
    while not terminated and step < 1000:

        if state[1] > 0:
            action = 2
        else:
            action = 0
        
        # 행동 실행
        next_state, reward, terminated, truncated, info = env.step(action)
        max_score = max(max_score, next_state[0])

        state = next_state
        step += 1
    
    print("Episode: {}, Steps: {}, Max Score: {}".format(episode, step, max_score))

env.close()
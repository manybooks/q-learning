import numpy as np
import gym
import random
import sys
import time

DEFAULT_EPISODE_NUM = 50000
DEFAULT_Q_VALUE = 0.0

if __name__ == "__main__":
    env = gym.make("Taxi-v2")

    action_size = env.action_space.n
    state_size = env.observation_space.n
    qtable = np.zeros((state_size, action_size))

    total_episodes = DEFAULT_EPISODE_NUM if len(sys.argv) == 1 else int(sys.argv[1])
    total_test_episodes = 100
    max_steps = 99

    learning_rate = 0.7
    gamma = 0.618

    epsilon = 1.0
    max_epsilon = 1.0
    min_epsilon = 0.01
    decay_rate = 0.01

    for episode in range(total_episodes):
        # Reset the environment
        state = env.reset()
        step = 0
        done = False
        
        for step in range(max_steps):
            exp_exp_tradeoff = random.uniform(0,1)
            if exp_exp_tradeoff > epsilon:
                action = np.argmax(qtable[state,:])
            else:
                action = env.action_space.sample()
            new_state, reward, done, info = env.step(action)
            qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * 
                                        np.max(qtable[new_state, :]) - qtable[state, action])
            state = new_state
            if done == True: 
                break
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 
    env.reset()
    rewards = []

    for episode in range(total_test_episodes):
        state = env.reset()
        step = 0
        done = False
        total_rewards = 0

        for step in range(max_steps):
            env.render()
            time.sleep(1)
            action_with_largest_q_value = np.argmax(qtable[state,:])
            largest_q_value = np.max(qtable[state,:])
            action = action_with_largest_q_value if largest_q_value != DEFAULT_Q_VALUE else random.randint(0, env.action_space.n - 1)

            new_state, reward, done, info = env.step(action)
            
            total_rewards += reward
            
            if done:
                rewards.append(total_rewards)
                break
            state = new_state
    env.close()
    print ("Score over time: " +  str(sum(rewards)/total_test_episodes))
import numpy as np
import gym
import random
import time
import sys


def train(total_episodes=100000):
    env = gym.make("Taxi-v2")
    
    action_size = env.action_space.n
    state_size = env.observation_space.n
    # qtable = np.zeros((state_size, action_size))
    qtable = np.empty((state_size, action_size))

    total_test_episodes = 100     # Total test episodes
    max_steps = 99                # Max steps per episode

    learning_rate = 0.7           # Learning rate
    gamma = 0.618                 # Discounting rate

    # Exploration parameters
    epsilon = 1.0                 # Exploration rate
    max_epsilon = 1.0             # Exploration probability at start
    min_epsilon = 0.01            # Minimum exploration probability 
    decay_rate = 0.01             # Exponential decay rate for exploration prob

    print("=========== 学習開始 ===========")
    for episode in range(total_episodes):
        if episode % 1000 == 0:
            print("エピソート{e}を学習中....".format(e=episode))
        # Reset the environment
        state = env.reset()
        step = 0
        done = False
        
        for step in range(max_steps):
            # 3. Choose an action a in the current world state (s)
            ## First we randomize a number
            exp_exp_tradeoff = random.uniform(0,1)
            
            ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
            if exp_exp_tradeoff > epsilon:
                action = np.argmax(qtable[state,:])
            
            # Else doing a random choice --> exploration
            else:
                action = env.action_space.sample()
            
            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, done, info = env.step(action)

            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * 
                                        np.max(qtable[new_state, :]) - qtable[state, action])
                    
            # Our new state is state
            state = new_state
            
            # If done : finish episode
            if done == True: 
                break
        
        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    
    return env, qtable

def test(env, qtable):
    state = env.reset()
    done = False
    max_steps = 99
    
    for step in range(max_steps):
        env.render()
        most_valuable_action = np.argmax(qtable[state, :])
        action = most_valuable_action if not np.isnan(most_valuable_action) else random.randint(0, env.action_space.n - 1)
        print(most_valuable_action)
        new_state, reward, done, info = env.step(action)
        if done:
            break
        state = new_state
        time.sleep(1)

if __name__ == "__main__":
    total_episodes = int(sys.argv[1])
    env, qtable = train(total_episodes)
    test(env, qtable)

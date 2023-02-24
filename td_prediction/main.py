import gym 
from collections import * 
import numpy as np

blackjack_env = gym.make("Blackjack-v1")

def td_prediction(env, gamma, alpha, noe, noi, train_policy_func): 
    v_table = defaultdict(float)
    action_space = [0, 1]

    for i in range(noi): 
        for e in range(noe): 
            trajectory = generate_episode(env, train_policy_func, v_table, action_space)
            T = len(trajectory) 
            for t in range(T-1): 
                St, At, Rt = trajectory[t] 
                St_1, At_1, Rt_1 = trajectory[t+1] 

                curr_val = v_table[St]
                td_target = Rt + gamma * v_table[St_1] 
                td_error = td_target - curr_val
                v_table[St] = curr_val + alpha * td_error

    return v_table


def train_policy_func(observation):
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1


def check_2d_array(observation): 
    if np.array(observation, dtype="object").shape == (2, ): 
        return True 
    return False 


def generate_episode(env, train_policy_func, v_table, action_space): 
    experience = []
    observation = env.reset()
    while True:
        if check_2d_array(observation=observation): 
            observation = observation[0]
        action = train_policy_func(observation)
        observation, reward, done, info, _ = env.step(action)
        experience.append((observation, action, reward))
        if done:
            break

    return experience 

v_table = td_prediction(blackjack_env, 1, 0.5, 10, 10, train_policy_func)
print(v_table)
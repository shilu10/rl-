# First Visit Monte-Carlo Control (incremental update)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import gym 
from collections import *
from random import *
from helper import *

blackjack_env = gym.make('Blackjack-v1')

def first_visit_mc_control(env, train_policy_func, epsilon, alpha, noe, noi):
    nb_actions = env.action_space.n
    action_space = [0, 1]
    q_table = defaultdict(float)
    for i in range(noi): 
        for e in range(noe):
            already_seen = defaultdict(bool) 
            trajectory = generate_episode(env, train_policy_func, q_table, epsilon, action_space)
            T = len(trajectory)
            G = 0
            for t in range(T-2, -1, -1):
                St, At, Rt = trajectory[t]
                St_1, At_1, Rt_1 = trajectory[t+1] 
                G = G + Rt_1

                if not already_seen.get(St, False): 
                    curr_q_value = q_table[(St, At)]
                    mc_target = G 
                    mc_error = alpha * (mc_target - curr_q_value)
                    q_table[(St, At)] = curr_q_value + mc_error
                    already_seen[St] = True 

    return q_table 

def generate_episode(env, train_policy_func, q_table, epsilon, action_space): 
    experience = []
    observation = env.reset()
    while True:
        if check_2d_array(observation=observation): 
            observation = observation[0]
        action = train_policy_func(epsilon, q_table, observation, action_space)
        observation, reward, done, info, _ = env.step(action)
        experience.append((observation, action, reward))
        if done:
            break

    return experience 

def train_policy_func(epsilon, q_table, St, action_space): 
    # epsilon -> probability of choosing random action
    # 1 -p -> probability of choosing already knewn action
    if np.random.rand() <= epsilon: 
        return np.random.choice(action_space)
    else: 
        return np.array([q_table[(St, a)] for a in action_space]).argmax()


def check_2d_array(observation): 
    if np.array(observation, dtype="object").shape == (2, ): 
        return True 
    return False 

def prediction_policy_func(q_table, St, action_space): 
    return np.array([q_table[(St, a)] for a in action_space]).argmax()

q_table = first_visit_mc_control(blackjack_env, train_policy_func, 0.10, 0.4, 10000, 10000)
plot_blackjack(q_table)
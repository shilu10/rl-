import gym
import numpy as np 
from policy import *


def initialize_q_table(nos_space, noa_space): 
    q_table = np.zeros((nos_space, noa_space))
    return q_table


def check_2d_array(observation): 
    if np.array(observation, dtype="object").shape == (2, ): 
        return True 
    return False 


def train_model(env, nos_space, noa_space, noe, max_epsilon, alpha, gamma, eps_decay_rate=0.005, min_epsilon=0.05): 
    action_space = [_ for _ in range(noa_space)] 
    q_table = initialize_q_table(nos_space, noa_space)

    games_reward = []
    for episode in range(noe): 
        state = env.reset()
        done = False 
        tot_rew = 0    
          
        if check_2d_array(state): 
            state = state[0]

        action = epsilon_greedy_policy(state, 
                                action_space, 
                                max_epsilon,
                                q_table,
                                env
                            )  

        if max_epsilon > min_epsilon:
            max_epsilon -= eps_decay_rate

        while not done: 
            next_state_info = env.step(action)
            next_state, reward_prob, done, info, _ = next_state_info

            next_action = epsilon_greedy_policy(next_state,
                                            action_space,
                                            max_epsilon,
                                            q_table,
                                            env
                                        )
            
            old_q_val = q_table[state][action]
            td_target = reward_prob + gamma * q_table[next_state][next_action]
            td_error = alpha * (td_target - old_q_val)
            new_q_val = old_q_val + td_error
            q_table[state][action] = new_q_val

            state = next_state
            action = next_action
            tot_rew += reward_prob

            if done: 
                games_reward.append(tot_rew)

        print(f"[+]Episode: {episode}, reward: {tot_rew}")

    return q_table, games_reward



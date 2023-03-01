import numpy as np

def epsilon_greedy_policy(curr_state, action_space, epsilon, q_table, env): 
    if np.random.rand() < epsilon: 
        explore = env.action_space.sample()
        return explore
    else: 
        exploit = np.argmax(q_table[curr_state])
        return exploit


def greedy_policy(curr_state, q_table): 
    greedy = q_table[curr_state].argmax()
    return greedy

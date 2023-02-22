import numpy as np 

class PolicyImprovement: 

    def __init__(self, env, number_of_state, action_space, discount): 
        self.action_space = action_space
        self.env = env 
        self.discount = discount
        self.number_of_state = number_of_state

    def greedy_improvement(self, value_table):
        policy = np.zeros(self.number_of_state)

        for state in range(self.number_of_state): 
            Q_table = np.zeros(self.action_space.n)
            for action in range(self.action_space.n):
                val = 0 
                for next_state_parameters in self.env.P[state][action]:
                    transition_prob, next_state, reward_prob, _ = next_state_parameters
                    val += (transition_prob * (reward_prob + self.discount * value_table[next_state]))
                Q_table[action] = val 
            policy[state] = np.argmax(Q_table)
        return policy

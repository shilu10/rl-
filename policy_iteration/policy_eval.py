import numpy as np

class PolicyEvaluation: 

    def __init__(self, value_table, action_space, env, state_space, theta, discount, number_of_states): 
        
        self.env = env
        self.action_space = action_space
        self.state_space = state_space
        self.theta = theta
        self.discount = discount
        self.number_of_states = number_of_states
        self.value_table = value_table

    def run(self, policy):  
        self.policy = policy
        delta = 0
        while True: 
            prev_value_table = np.copy(self.value_table)
            # Full Sweep
            for state in range(self.number_of_states): 
                action = self.policy[state] 
                value_for_curr_state = self.cal_curr_state_val(
                                                            self.env,
                                                            state,
                                                            action,
                                                            prev_value_table
                                                        )
                self.value_table[state] = value_for_curr_state
                delta = max(delta, abs(self.value_table[state] - prev_value_table[state]))

            if (np.sum(np.fabs(prev_value_table - self.value_table))<=self.theta):
                break

        return self.value_table
    
    def cal_curr_state_val(self, env, curr_state, action, prev_value_table): 

        value_for_curr_state = 0 
        for next_state_parameters in self.env.P[curr_state][action]: 
            transition_prob, next_state, reward_prob, isTerminate = next_state_parameters
            value_for_curr_state += transition_prob * (reward_prob + self.discount*prev_value_table[next_state])

        return value_for_curr_state









        
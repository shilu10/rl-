import numpy as np 
import gym 

frozen_lake_environment = gym.make("FrozenLake-v1")
state_space = frozen_lake_environment.observation_space
number_of_state_space = state_space.n 
action_space = frozen_lake_environment.action_space
number_of_action_space = action_space.n 
theta = 1e-20
discount = 0.9
number_of_iteration = 500


class ValueIteration: 
    def __init__(self, env, action_space, state_space, nos, noa, noi, gamma, theta): 
        self.env = env 
        self.action_space = action_space 
        self.state_space = state_space 
        self.nos = nos 
        self.noa = noa 
        self.gamma = gamma 
        self.theta = theta 
        self.noi = noi 

    def _val_iter(self, policy, value_table): 
        prev_value_table = np.copy(value_table)
        delta = 0
        while True: 
            for state in range(self.nos): 
                v = value_table[state]
                Q_value = []
                for action in range(self.noa): 
                    val = self.sum_cal(state, action, self.gamma, prev_value_table)
                    Q_value.append(sum(val)) 
                delta = max(delta, abs(v - value_table[state]))

            value_table[state] = max(Q_value)
            # Converage condition.
            if delta < self.theta:
                break

        return value_table

    def sum_cal(self, state, action, gamma, prev_vt): 
        next_states_rewards = [] 
        for next_state_info in self.env.P[state][action]: 
            trans_prob, next_state, reward_prob, _ = next_state_info
            val = trans_prob * (reward_prob + gamma * prev_vt[next_state])
            next_states_rewards.append(val)

        return next_states_rewards

    def greedy_policy_improvement(self, value_table): 
        policy = np.zeros(self.nos) 
        for state in range(self.nos):
            Q_table = np.zeros(self.noa)
            for action in range(self.noa):
                for next_sr in self.env.P[state][action]: 
                    trans_prob, next_state, reward_prob, _ = next_sr 
                    Q_table[action] += (trans_prob * (reward_prob + self.gamma * value_table[next_state]))
            policy[state] = np.argmax(Q_table)

        return policy 

    def run(self):
        random_policy = np.zeros(number_of_action_space, dtype="int")
        value_table = np.zeros(number_of_state_space, dtype="int")
        for _ in range(self.noi): 
            value_table = self._val_iter(random_policy, value_table)
            random_policy = self.greedy_policy_improvement(value_table)

        return random_policy, value_table


vi = ValueIteration(
    frozen_lake_environment,
    action_space,
    state_space,
    number_of_state_space,
    number_of_action_space,
    number_of_iteration,
    discount,
    theta
)

print(vi.run())
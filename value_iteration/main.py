import numpy as np 
import gym 
from eval import *

frozen_lake_environment = gym.make("FrozenLake-v1")
state_space = frozen_lake_environment.observation_space
number_of_state_space = state_space.n 
action_space = frozen_lake_environment.action_space
number_of_action_space = action_space.n 
theta = 1e-20
discount = 1
number_of_iteration = 10000


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
                    Q_value.append(val) 

                value_table[state] = max(Q_value)
                delta = max(delta, abs(v - value_table[state]))
            # Converage condition.
            if (np.sum(np.fabs(prev_value_table - value_table))<=self.theta):
                break

        return value_table

    def sum_cal(self, state, action, gamma, prev_vt): 
        next_states_rewards = 0
        for next_state_info in self.env.P[state][action]: 
            trans_prob, next_state, reward_prob, _ = next_state_info
            next_states_rewards += trans_prob * (reward_prob + gamma * prev_vt[next_state])
        
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

    def check(self, learned_policy, learned_vf): 
        a2w = {0:'<', 1:'v', 2:'>', 3:'^'}
        policy_arrows = np.array([a2w[x] for x in learned_policy])

        correct_vf = np.array([[0.82352941, 0.82352941, 0.82352941, 0.82352941],
                            [0.82352941, 0.        , 0.52941176, 0.        ],
                            [0.82352941, 0.82352941, 0.76470588, 0.        ],
                            [0.        , 0.88235294, 0.94117647, 0.        ]])
        correct_policy_arrows = np.array([['<', '^', '^', '^'],
                                  ['<', '<', '<', '<'],
                                  ['^', 'v', '<', '<'],
                                  ['<', '>', 'v', '<']])

        if (np.allclose(learned_vf.reshape([4,-1]), correct_vf) 
                and np.alltrue(policy_arrows.reshape([4,-1]) == correct_policy_arrows)): 
                print("It is actually a Optimal Policy")

        else: 
            print("Not a Optimal Policy")


value_iteration = ValueIteration(
    frozen_lake_environment,
    action_space,
    state_space,
    number_of_state_space,
    number_of_action_space,
    number_of_iteration,
    discount,
    theta
)

learned_policy, learned_value_function = value_iteration.run()
value_iteration.check(learned_policy, learned_value_function)

mean_rwd, std_rwd = eval_model(frozen_lake_environment, 200, learned_policy, None)

print(f"Mean Reward: {mean_rwd}, Std Reward: {std_rwd}")
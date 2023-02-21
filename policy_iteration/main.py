import gym 
from utils import *
import numpy as np
from policy_eval import * 
from policy_improvement import *

frozen_lake_env = gym.make('FrozenLake-v1') 
action_space = frozen_lake_env.action_space
state_space = frozen_lake_env.observation_space
number_of_states = frozen_lake_env.observation_space.n
number_of_actions = frozen_lake_env.action_space.n

# Initial Random Policy and Value for each state(s) in state space(S). 
random_value_table = np.zeros(number_of_states)
random_policy = np.zeros(number_of_states)
number_of_iteration = 20000

# Reseting and Rendering again
frozen_lake_env.reset()
frozen_lake_env.render()

class PolicyIteration: 

    def __init__(self, env, action_space, state_space, value_table, number_of_states, policy, discount, theta):
        self.env = env 
        self.action_space = action_space 
        self.state_space = state_space
        self.value_table = value_table 
        self.number_of_states = number_of_states  
        self.policy = policy
        self.discount = discount
        self.theta = theta

        self.policy_evaluation = PolicyEvaluation(
            self.policy,
            self.action_space,
            self.env,
            self.state_space,
            self.theta,
            self.discount,
            self.number_of_states
        )

        self.policy_improvement = PolicyImprovement(
            self.env,
            self.number_of_states,
            self.action_space, 
            self.discount
        )

    def run(self, number_of_iteration): 
        policy = self.policy
        for _ in range(number_of_iteration): 
            value_table = self.policy_evaluation.run(policy) 
            new_policy = self.policy_improvement.greedy_improvement(value_table)

            if np.all(new_policy == self.policy): 
                break 
            
            policy = new_policy
        self.policy = policy
        return self.policy

policy_iteraton = PolicyIteration(
    frozen_lake_env, 
    action_space,
    state_space,
    random_value_table,
    number_of_states,
    random_policy,
    1,
    1e-20, 
)
print(policy_iteraton.run(number_of_iteration))


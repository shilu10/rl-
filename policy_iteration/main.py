import gym
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
number_of_iteration = 2100

# Reseting and Rendering again
frozen_lake_env.reset()
frozen_lake_env.render()

class PolicyIteration: 
    """
        run method: 
                -> This method contains the code to run the policy iteration by combining step 
                    of both the policy evaluation and policy improvement.
                -> Input: Number of Iteration
                -> Output: Learned Policy(Target Policy), Learned Value Function(Target Value Function)

        check method: 
                -> This metod contains the code to checking whether the Learned Policy is 
                    actually a Optimal Policy for the environment or not.
                -> Input: Learned Policy, Learned Value Function.
                -> Output: Bool(Learning Policy == Optimal Policy)
    """

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
        return self.policy, value_table

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

learned_policy, learned_vf = policy_iteraton.run(number_of_iteration)
policy_iteraton.check(learned_policy, learned_vf)



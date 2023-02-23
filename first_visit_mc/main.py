import numpy
import gym
from matplotlib import pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from functools import partial
from generate_episode import * 

plt.style.use('ggplot')

blackjack_env = gym.make('Blackjack-v1')

class FirstVisitMonteCarloPrediction:

    def __init__(self, env, noe, gamma): 
        self.env = env
        self.noe = noe  
        self.gamma = gamma
        self.generate_episode = GenerateEpisode()

    def run(self): 
        value_table = defaultdict(float)
        seen_at_current = defaultdict(bool)
        N = defaultdict(int)
        

        for episode in range(self.noe): 
            self.env.reset()
            trajectory = self.generate_episode.run(self.env, self.sample_policy)
            G = 0

            for i in range(len(trajectory)-1, -1, -1):    
                state, reward, action = trajectory[i]
                reward_1 = 0
                if i+1 <= len(trajectory)-1: 
                    state_1, reward_1, action = trajectory[i+1]
                G = self.gamma * G + reward_1
                print(reward)
                if not seen_at_current.get(state, False):
                    seen_at_current[state] = True  
                    value_table[state] += G
                    N[state] += 1 
            seen_at_current = defaultdict(dict)

        for state in value_table.keys(): 
            value_table[state] = value_table[state] / N[state]
        return value_table, p

    def sample_policy(self, observation):
        score, dealer_score, usable_ace = observation
        return 0 if score >= 20 else 1

m = FirstVisitMonteCarloPrediction(blackjack_env, 100, 0.9)
m.run()


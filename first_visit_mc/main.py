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

    def __init__(self, env, noe): 
        self.env = env
        self.noe = noe  
        self.generate_episode = GenerateEpisode()

    def run(self): 
        value_table = defaultdict(float)
        N = defaultdict(int)

        for episode in range(self.noe): 
            print(self.noe, episode)
            self.env.reset()
            trajectory = self.generate_episode.run(self.env, self.sample_policy)
            returns = 0
            print(len(trajectory))
            for i, observation in enumerate(reversed(trajectory)):    
                state, reward, action = observation
                print(observation)
                returns += reward
                if N.get(state, 0) == episode: 
                    print(state, "in")
                    N[state] += 1 
                    value_table[state] += returns

        for state in value_table.keys(): 
            value_table[state] = value_table[state] / N[state]
        return value_table

    def sample_policy(self, observation):
        score, dealer_score, usable_ace = observation
        return 0 if score >= 20 else 1

m = FirstVisitMonteCarloPrediction(blackjack_env, 100)
print(m.run())


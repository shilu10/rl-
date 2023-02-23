import numpy as np

class GenerateEpisode: 

    def __init__(self): 
        pass 

    def check_2d_obser(self, observation): 
        if np.array(observation, dtype="object").shape == (2, ): 
            return True 
        return False 
    
    def run(self, env, policy):
        trajectory = []
        done = True
        while True:
            if done:
                observation, reward, done = env.reset(), None, False
            else:
                observation, reward, done, info, _ = env.step(action)
            if self.check_2d_obser(observation): 
                observation = observation[0]
            action = policy(observation)
            trajectory.append((observation, action, reward))
            if done:
                break

        return trajectory